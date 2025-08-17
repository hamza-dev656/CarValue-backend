# scripts/data_export.py
"""
Data export utilities for training data preparation.
Handles streaming data from repository to TSV files for CatBoost training.
"""

import os
import csv
import time
import logging
from typing import Optional, Iterable, Dict, Tuple

import numpy as np

from app.repositories.vehicle_repository import VehicleRepository
from app.utils.constants import TRAIN_COLS
from app.services.bootstrap.data_processing import state_to_region, to_float
from app.services.training.model_utils import write_catboost_column_description

log = logging.getLogger("data_export")

from app.config.model_config import DATA_DIR, REUSE_TSV, FORCE_REEXPORT


def iter_training_rows_global(limit_per_mm: int = 15000) -> Iterable[Dict]:
    """
    Stream rows from repository, capping per (make, model).
    VehicleRepository.stream_training_rows() yield ORM rows in chunks.
    """
    counts: Dict[Tuple[str, str], int] = {}
    total = 0
    for v in VehicleRepository.stream_training_rows():
        if v.listing_price is None:
            continue
        price = float(v.listing_price)
        if price < 500 or price > 200_000:
            continue

        make = (v.make or "UNK")
        model = (v.model or "UNK")
        key = (make, model)
        if counts.get(key, 0) >= limit_per_mm:
            continue

        row = {
            "log_price": float(np.log1p(price)),
            "year": int(v.year) if v.year is not None else None,
            "make": str(make),
            "model": str(model),
            "trim": str(v.trim or "UNK"),
            "listing_mileage": to_float(v.listing_mileage),
            "dealer_state": str(v.dealer_state or "UNK"),
            "exterior_color": str(v.exterior_color or "UNK"),
            "style": str(v.style or "UNK"),
            "driven_wheels": str(v.driven_wheels or "UNK"),
            "fuel_type": str(v.fuel_type or "UNK"),
            "interior_color": str(v.interior_color or "UNK"),
            "used": "YES" if getattr(v, "used", False) else "NO",
            "certified": "YES" if getattr(v, "certified", False) else "NO",
            "listing_status": str(getattr(v, "listing_status", "UNK")),
            "region": state_to_region(getattr(v, "dealer_state", None)),
            "last_seen_date": str(getattr(v, "last_seen_date", "") or ""),
        }
        counts[key] = counts.get(key, 0) + 1
        total += 1
        if total % 200_000 == 0:
            log.info(f"[GLOBAL] streamed {total:,} rows (unique YMM capped at {limit_per_mm} each)")
        yield row


def stream_export_tsv(rows: Iterable[Dict], base_path: str, holdout_every: int = 5) -> Tuple[str, str, int, int]:
    """
    Write train/valid TSVs streaming; every Nth row goes to validation.
    Returns train_path, valid_path, n_train, n_valid.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    train_path = base_path.replace(".tsv", "_train.tsv")
    valid_path = base_path.replace(".tsv", "_valid.tsv")
    n_tr = n_va = 0
    t0 = time.time()

    with open(train_path, "w", newline="", encoding="utf-8") as ftr, \
         open(valid_path, "w", newline="", encoding="utf-8") as fva:
        wtr = csv.writer(ftr, delimiter="\t")
        wva = csv.writer(fva, delimiter="\t")
        wtr.writerow(TRAIN_COLS)
        wva.writerow(TRAIN_COLS)
        for i, r in enumerate(rows, start=1):
            row = [
                r.get("log_price"), r.get("year"), r.get("make"), r.get("model"),
                r.get("trim"), r.get("listing_mileage"), r.get("dealer_state"),
                r.get("exterior_color"), r.get("style"), r.get("driven_wheels"),
                r.get("fuel_type"), r.get("interior_color"), r.get("used"),
                r.get("certified"), r.get("listing_status"), r.get("region"),
                r.get("last_seen_date"),
            ]
            if (i % holdout_every) == 0:
                wva.writerow(row)
                n_va += 1
            else:
                wtr.writerow(row)
                n_tr += 1
            if i % 200_000 == 0:
                log.info(f"[EXPORT] wrote {i:,} rows …")

    log.info(f"[EXPORT] done: train={n_tr:,} valid={n_va:,} in {time.time()-t0:.1f}s")
    return train_path, valid_path, n_tr, n_va


def tsv_base(data_version: str = None) -> str:
    """Generate base TSV path - simplified to not use version"""
    return os.path.join(DATA_DIR, "global_training.tsv")


def tsv_paths(data_version: str = None) -> Tuple[str, str, str]:
    """Generate train, valid, and column description file paths - simplified to not use version"""
    base = tsv_base()
    return (
        base.replace(".tsv", "_train.tsv"),
        base.replace(".tsv", "_valid.tsv"),
        base.replace(".tsv", ".cdesc")
    )


def file_ok(path: str) -> bool:
    """Check if file exists and has content"""
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def count_rows_quick(tsv_path: str) -> int:
    """Quickly count rows in TSV file (excluding header)"""
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def prepare_training_data(data_version: str = None, limit_per_mm: int = 15000) -> Tuple[str, str, str, int, int]:
    """
    Prepare training data for global model training.
    Handles TSV export, reuse logic, and column descriptions.
    Now simplified to not use version-based filenames.
    
    Returns:
        Tuple of (train_path, valid_path, cdesc_path, n_train, n_valid)
    """
    log.info(f"[DATA PREP] Starting data preparation (limit_per_mm={limit_per_mm})")
    log.info(f"[DATA PREP] Configuration: reuse_tsv={REUSE_TSV}, force_reexport={FORCE_REEXPORT}")
    
    train_path, valid_path, cdesc_path = tsv_paths()
    
    # Ensure column description exists
    log.info(f"[DATA PREP] Writing column description: {cdesc_path}")
    write_catboost_column_description(cdesc_path)
    
    # Check if we need to export fresh data
    need_export = FORCE_REEXPORT or (not (file_ok(train_path) and file_ok(valid_path)))
    
    if REUSE_TSV and not FORCE_REEXPORT and not need_export:
        # Reuse existing TSVs
        n_train = count_rows_quick(train_path)
        n_valid = count_rows_quick(valid_path)
        log.info(f"[DATA PREP] Reusing existing TSVs:")
        log.info(f"  - Train: {train_path} ({n_train:,} rows)")
        log.info(f"  - Valid: {valid_path} ({n_valid:,} rows)")
    else:
        # Export fresh TSVs
        base_path = tsv_base()
        log.info(f"[DATA PREP] Exporting fresh TSVs to: {base_path}")
        
        train_path, valid_path, n_train, n_valid = stream_export_tsv(
            iter_training_rows_global(limit_per_mm), 
            base_path, 
            holdout_every=5
        )
        log.info(f"[DATA PREP] Export complete:")
        log.info(f"  - Train: {train_path} ({n_train:,} rows)")
        log.info(f"  - Valid: {valid_path} ({n_valid:,} rows)")
    
    log.info(f"[DATA PREP] Data preparation complete")
    return train_path, valid_path, cdesc_path, n_train, n_valid
