# app/services/bootstrap.py
import os
import re
import glob, threading, logging
from datetime import datetime
from typing import Optional

from app.services.training.training_service import TrainingService
from app.services.model_registry.model_registry import ModelRegistry, ModelMeta
from app import cache
from app.config.model_config import ARTIFACT_DIR, DATA_VERSION

# -------- Logging --------
log = logging.getLogger("bootstrap")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s bootstrap: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

TRAINING_LOCK_KEY = "ml:global_training:in_progress"
TRAINING_LOCK_TTL = int(os.getenv("GLOBAL_TRAINING_LOCK_TTL", str(2 * 60)))  # 60 minutes

def _default_data_version() -> str:
    return DATA_VERSION if DATA_VERSION != "v00000000" else datetime.utcnow().strftime("v%Y%m%d")

def _train_global_job(app, data_version: Optional[str] = None, limit_per_mm: int = None):
    with app.app_context():
        try:
            dv = data_version or _default_data_version()
            limit = limit_per_mm or int(os.getenv("GLOBAL_LIMIT_PER_MM", "15000"))
            registry = ModelRegistry(cache)

            # If meta exists, skip
            meta = registry.get_global_meta()
            if meta:
                log.info(f"Global model exists (rmse_pct={meta.rmse_pct:.2f}%, version={meta.data_version})")
                return

            # If artifact exists on disk (cache cleared), skip
            artifacts = glob.glob(os.path.join(ARTIFACT_DIR, f"global_{dv}.cbm"))
            if artifacts:
                log.info(f"Found artifact {artifacts[0]} — skipping training")
                return

            log.info(f"Starting global training (data_version={dv}, limit_per_mm={limit}) …")
            meta = TrainingService.train_global(dv, limit_per_mm=limit)
            if meta:
                log.info(f"Global model trained | rmse_pct={meta.rmse_pct:.2f}% | n_train={meta.n_train:,} | path={meta.path}")
            else:
                log.error("Global training returned no meta (failed).")
        except Exception as e:
            log.exception(f"Error during global training: {e}")
        finally:
            cache.delete(TRAINING_LOCK_KEY)

def initialize_models_async(app):
    """
    Start background global training once per process.
    Uses cache lock to avoid duplicates across workers/reloaders.
    """
    registry = ModelRegistry(cache)
    
    # Start periodic cleanup of expired model files
    try:
        registry.schedule_periodic_cleanup(interval_hours=24)
        log.info("Scheduled periodic model cleanup (every 24 hours)")
    except Exception as e:
        log.error(f"Failed to schedule model cleanup: {e}")

    # If already registered, nothing to do
    meta = registry.get_global_meta()
    if meta:
        log.info(f"Global model already exists (rmse_pct={meta.rmse_pct:.2f}%, version={meta.data_version})")
        return

    # If artifact exists (cache cleared scenario), skip
    dv = _default_data_version()
    artifacts = glob.glob(os.path.join(ARTIFACT_DIR, "global_*.cbm"))
    if artifacts:
        log.info(f"Global model file present: {artifacts[0]} (skipping bootstrap)")
        return

    # Acquire lock
    if cache.get(TRAINING_LOCK_KEY):
        log.info("Global training already in progress on another worker.")
        return
    cache.set(TRAINING_LOCK_KEY, "1", timeout=TRAINING_LOCK_TTL)

    log.info("Scheduling global training in background thread …")
    t = threading.Thread(target=_train_global_job, args=(app, dv), daemon=True)
    t.start()
