# VinAudit CarValue — Backend System Design (Revised)

*Last updated: Aug 2025*

## 0) Executive Summary

VinAudit CarValue is a Flask-based REST API that computes market-price estimates for vehicles from a large historical listings corpus. It serves prices and comps in real time using a **hybrid modeling strategy**:

1. **Local CatBoost models** (per Year–Make–Model) for high-traffic segments → highest accuracy.
2. **Global CatBoost model** for broad coverage.
3. **Statistical fallback** (simple / adjusted average) for cold starts.

The system is optimized for **4–5M rows** via streaming exports, on-disk training datasets (TSV), lean column descriptions for CatBoost, and background training with cache-based locks.

---

## 1) High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                               API Clients                                  │
│                 (Next.js, internal tools, integrations)                     │
└────────────────────────────────────────────────────────────────────────────┘
                 │  HTTPS                                                     
                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                 Flask API                                   │
│  Controllers: /api/search, /api/makes-models                                │
│  Service layer: VehicleService                                              │
│  Caching: Flask-Caching (Redis)                                             │
└────────────────────────────────────────────────────────────────────────────┘
      │                         │                               │
      │ (SQLAlchemy)           │ (Redis)                        │ (Disk)
      ▼                         ▼                               ▼
┌───────────────┐     ┌──────────────────────┐       ┌────────────────────────┐
│ PostgreSQL    │     │ Redis (shared)       │       │ Artifacts (ARTIFACT_DIR)│
│ vehicles tbl  │     │ • hit counters       │       │ • global_*.cbm         │
│ + indexes     │     │ • training locks     │       │ • local_*.cbm          │
└───────────────┘     │ • model metadata JSON│       │ • global_*_train.tsv   │
                      └──────────────────────┘       │ • global_*_valid.tsv   │
                                                     └────────────────────────┘
                                       ▲
                                       │ ModelRegistry (load/save + metadata)
                                       │
                          ┌────────────────────────────────────┐
                          │        ML Training Subsystem       │
                          │  • Global trainer (TSV-based)      │
                          │  • Local trainer (DB streaming)    │
                          │  • Background threads or LTS       │
                          └────────────────────────────────────┘
```

**Key flows**

* **Serving:** VehicleService → ModelRegistry → load local/global `.cbm` → preprocess request row → predict → comps via repository → response.
* **Global training:** export TSVs once (reuse if present) + column description → CatBoost fit with early stopping → save `.cbm` + register metadata in Redis.
* **Local training:** hit-counter triggers background job → stream Y/M/M rows from DB, same preprocessing, CatBoost fit → save `.cbm` + register.

---

## 2) Core Features

* **Price estimate** rounded to nearest \$100.
* **Comparable listings** (up to 100) used to contextualize the estimate.
* **Confidence metrics** (RMSE% tiers → high/medium/low).
* **Hybrid model selection** (local → global → fallback) with transparent `method` field.
* **Hit-based auto-training** for popular Y/M/M segments.
* **Background training** with locks to prevent duplicate jobs.
* **Artifact & metadata registry** (disk `.cbm` + Redis JSON pointer).
* **TSV reuse** for global training (fast restarts, low RAM).

---

## 3) API Design

### 3.1 Endpoints

**GET `/api/search`**
Query params: `year` (int), `make` (str), `model` (str), optional `mileage` (int), `trim`, `color`, `dealer_state`.

**Response (example)**

```json
{
  "estimate": "$13,800",
  "listings": [ { /* up to 100 comps */ } ],
  "calculation_date": "2025-08-15T02:22:21Z",
  "method": "local_model | global_model | simple_average",
  "model_accuracy": {
    "rmse": null,
    "rmse_percentage": 15.0,
    "confidence": "medium"
  },
  "model_meta": {
    "kind": "global",
    "artifact": "global_v20250814.cbm",
    "data_version": "v20250814"
  }
}
```

**GET `/api/makes-models`**
Returns nested `year → make → [models]` for UI pickers.

### 3.2 Errors

* `400` invalid parameter types / missing required fields.
* `500` internal error (wrapped & logged).

---

## 4) Data Model (PostgreSQL)

```sql
CREATE TABLE vehicles (
  id                BIGSERIAL PRIMARY KEY,
  vin               VARCHAR(17) UNIQUE NOT NULL,
  year              INT NOT NULL,
  make              VARCHAR(50) NOT NULL,
  model             VARCHAR(50) NOT NULL,
  trim              VARCHAR(100),
  dealer_name       VARCHAR(100),
  dealer_street     VARCHAR(100),
  dealer_city       VARCHAR(50),
  dealer_state      VARCHAR(20),
  dealer_zip        VARCHAR(10),
  listing_price     NUMERIC(10,2),
  listing_mileage   INT,
  used              BOOLEAN DEFAULT TRUE,
  certified         BOOLEAN DEFAULT FALSE,
  style             VARCHAR(100),
  driven_wheels     VARCHAR(50),
  engine            VARCHAR(100),
  fuel_type         VARCHAR(50),
  exterior_color    VARCHAR(50),
  interior_color    VARCHAR(50),
  seller_website    VARCHAR(255),
  first_seen_date   TIMESTAMPTZ,
  last_seen_date    TIMESTAMPTZ,
  dealer_vdp_last_seen_date TIMESTAMPTZ,
  listing_status    VARCHAR(50)
);

-- High-value indexes
CREATE INDEX idx_year_make_model ON vehicles (year, make, model);
CREATE INDEX idx_make_model_year ON vehicles (make, model, year);
CREATE INDEX idx_price_mileage   ON vehicles (listing_price, listing_mileage);
-- Helpful partials (optional)
CREATE INDEX IF NOT EXISTS idx_mm_nonnull_price ON vehicles (make, model) WHERE listing_price IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_y_mmile ON vehicles (year, listing_mileage) WHERE listing_mileage IS NOT NULL;
```

**Repository queries**

* `get_listings_with_filters({year, make, model, listing_price=True}) LIMIT 100 ORDER BY listing_mileage`
* `get_average_price_with_filters(...)`

---

## 5) Model Lifecycle

### 5.1 Global training (TSV-based)

1. **prepare\_training\_data(data\_version, limit\_per\_mm)**

   * Streams from DB per (make, model) up to `limit_per_mm` rows.
   * Cleans + derives features, writes `*_train.tsv` / `*_valid.tsv` and `*.cdesc` (column description).
   * Reuses existing TSVs if present (fast warm start).
2. **CatBoost fit** with early stopping; predict in **log-price** space, evaluate RMSE% in price space.
3. **Save artifact** `global_<dv>.cbm` → disk; compute SHA256 & size.
4. **Register metadata** in Redis (`model_registry:global`).

### 5.2 Local training (DB streaming)

1. **Trigger**: hit counter ≥ threshold for a Y/M/M.
2. **Lock**: acquire `lock:local:<y>:<make>:<model>` in Redis.
3. **Data**: stream rows for that Y/M/M (cap e.g. 50k), **same preprocessing** as global.
4. **Fit** a compact CatBoost model; time-based split if `last_seen_date` exists.
5. **Save** `local_<y>_<make>_<model>_<dv>.cbm` and **register metadata** (`model_registry:local:<seg>`).
6. **Release** lock regardless of outcome.

### 5.3 Serving-time selection

```
try local meta → load .cbm → predict
else try global meta → (bump hits, maybe enqueue local) → predict
else fallback average
```

---

## 6) Preprocessing (Train & Serve Parity)

**Cleaning**

* Numeric coercion for price/mileage; drop rows with missing price; keep mileage (may be imputed).
* Bounds: price ∈ \[500, 200,000].

**Outlier control (winsorization)**

* Per (make, model) clip price & mileage to quantiles (e.g., 0.5%–99.5%).
* For mileage, only clip when not null.

**Feature engineering**

* `age = current_year - year` (0–40),
* mileage **imputation** (`age * 12,000` fallback; flag `mileage_imputed`),
* `log_mileage = log1p(mileage)`,
* `region` from `dealer_state`,
* trim normalization → common buckets (`XLE`, `LE`, `LIMITED`, ...),
* booleans → `"YES"/"NO"`, strings for all categoricals.

**Target**

* Train on `log_price = log1p(listing_price)`; at serve time: `price = expm1(pred)`.

**Serve-time row**

* Build a 1-row DataFrame; run the same derivation helper; predict with CatBoost.

---

## 7) Caching, Counters, and Locks

* **Prediction cache**: memoize responses for (y, make, model, options) \~10 min.
* **Hit counters**: Redis key `hits:<y>:<make>:<model>` with TTL (e.g., 24h). When ≥ threshold, call `enqueue_local_training(...)`.
* **Training locks**: key `lock:local:<seg>` with TTL > expected training time to avoid duplicate workers. Always released in `finally`.

**Background Training**

* Prefer offloading to **Local Training Service** via HTTP POST (short timeout); fallback to **daemon thread** with `app.app_context()`.

---

## 8) Configuration

| Key                   | Purpose                             | Example                    |
| --------------------- | ----------------------------------- | -------------------------- |
| `DATABASE_URL`        | Postgres connection                 | `postgres://...`           |
| `REDIS_URL`           | Cache/locks/hits                    | `redis://localhost:6379/0` |
| `ARTIFACT_DIR`        | Model files                         | `./artifacts`              |
| `GLOBAL_LIMIT_PER_MM` | Cap rows per (make,model) in export | `15000`                    |
| `EARLY_STOP`          | CatBoost early stopping rounds      | `100`                      |
| `HITS_THRESHOLD`      | Trigger local training              | `20`                       |
| `TRAINING_LOCK_TTL`   | Lock expiry seconds                 | `3600`                     |
| `LOCAL_MAX_ROWS`      | Cap rows for local training         | `50000`                    |
| `LOCAL_TRAINER_URL`   | External trainer base URL           | `http://trainer:8080`      |
| `REUSE_TSV`           | Reuse existing global TSVs          | `true`                     |
| `FORCE_REEXPORT`      | Force rebuild TSVs                  | `false`                    |

---

## 9) Logging & Observability

* **Structured logs** with component prefixes: `training`, `bootstrap`, `vehicle`, `registry`.
* CatBoost `verbose` progress every N trees.
* Key events: TSV reuse vs export, pool creation, fit start/stop, best iteration, RMSE%, artifact saved (size/sha), model registered, lock acquire/release, enqueue results.
* Consider metrics: request latency, cache hit rate, training duration, queue depth.

**Example snippets**

```
[2025-08-15 02:18:35] INFO training: [GLOBAL] CatBoost fit() …
200: learn: 0.2623 test: 0.2607 best: 0.2607 (200)
[2025-08-15 02:22:21] INFO training: [GLOBAL] saved ./artifacts/global_v20250814.cbm (40.7MB) sha256=8c5050de6a34…
```

---

## 10) Performance Notes

* **Global training** on \~3.5M rows (train+valid after caps) completes in minutes when reusing TSVs; RAM bounded by CatBoost’s `used_ram_limit` and DSV reader.
* **Local training** typically < 50k rows: seconds to a few minutes.
* **Serving latency**: single prediction from loaded model is milliseconds; total API often < 200ms including DB comps.

DB tips: `ANALYZE` after bulk loads; periodic `VACUUM`; ensure composite indexes are used in query plans (watch `ORDER BY listing_mileage LIMIT 100`).

---

## 11) Testing Strategy

* **Unit**: preprocessing helpers, winsorization, trim normalization, region mapping.
* **Integration**: search endpoint happy-path, error cases, global present vs absent, hit-counter threshold path.
* **Training**: smoke test on tiny fixtures (TSV + DB); assert artifact exists and meta registered; assert RMSE% finite.
* **Load**: curl/xargs to exercise parallel requests; verify locks prevent duplicate training.

---

## 12) Security & Compliance

* Input validation + type coercion; ORM-bound parameters avoid SQL injection.
* CORS restricted by environment; consider API keys/rate-limits for public endpoints.
* Secrets via environment; artifacts are non-PII.

---

## 13) Operations Runbook

**Startup**

1. App boots; `bootstrap.initialize_models_async(app)` checks cache; if missing but artifacts exist, registers latest global model; may kick off global training.
2. Health-check: `/api/search` with a known Y/M/M.

**Common Tasks**

* **Retrain global**: delete TSVs or set `FORCE_REEXPORT=true`; restart to re-export & fit.
* **Reset local for Y/M/M**: delete `model_registry:local:<seg>` + file; next hits retrigger training.
* **Clear stale locks**: locks auto-expire; manual delete if needed.

**Failure modes**

* **Cache lost**: rehydrate by scanning `ARTIFACT_DIR` and re-registering latest models.
* **TSV schema mismatch**: regenerate `*.cdesc` and TSVs.

---

## 14) Roadmap

* Better **regional adjustments** (ZIP-level), seasonality.
* Feature store for durable, versioned features.
* Move background work to Celery/RQ; add retries & backoff.
* Model history & canary deployments (switchable aliases).
* Quality dashboards: RMSE% drift by segment + traffic heatmap.

---

## 15) Appendix: Module Map

```
app/
├─ controllers/
│  └─ vehicle_controller.py           # REST endpoints
├─ repositories/
│  └─ vehicle_repository.py           # DB access
├─ services/
│  ├─ vehicle_service.py              # Serve-time orchestration
│  ├─ training_service.py             # Global + local trainers
│  ├─ bootstrap.py                    # Startup checks + async global training
│  └─ model_registry.py               # ModelMeta + load/save + cache registry
├─ utils/
│  ├─ training_utils.py               # Enqueue local training (LTS/thread)
│  ├─ cache_utils.py                  # Locks, hit counters
│  ├─ data_processing.py              # Preprocessing helpers (derive/clip)
│  ├─ model_utils.py                  # CatBoost builders, cdesc writer
│  └─ config.py                       # ENV + constants
└─ artifacts/                         # *.cbm, *_train.tsv, *_valid.tsv
```
