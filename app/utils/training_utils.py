# app/utils/training_utils.py
"""
Training utilities and background job management
"""

# ── Standard library imports ───────────────────────────────────────────────────
import logging
import os
import threading

# ── Third-party imports ────────────────────────────────────────────────────────
import requests

# ── Local imports ──────────────────────────────────────────────────────────────
from app.config.model_config import DATA_VERSION, LOCAL_TRAINER_URL
from .cache_utils import acquire_training_lock, release_training_lock

log = logging.getLogger(__name__)


def enqueue_local_training(cache, year: int, make: str, model_name: str,
                           data_version: str = None) -> None:
    """
    Enqueue local training for the given year/make/model.
    Prefers external training service, falls back to background thread.
    """
    # Import kept inside to avoid potential circular imports during module load.

    try:
        # Check if already training
        if not acquire_training_lock(cache, year, make, model_name):
            log.info(f"[LOCAL {year}-{make}-{model_name}] training already in progress (locked)")
            return

        log.info(f"[LOCAL {year}-{make}-{model_name}] hit threshold reached, starting training...")

        # Prefer calling a separate Local Training Service (non-blocking, short timeout)
        if LOCAL_TRAINER_URL:
            payload = {
                "year": year,
                "make": make,
                "model": model_name,
                "data_version": data_version or DATA_VERSION,
                "force": False,
            }
            try:
                requests.post(
                    f"{LOCAL_TRAINER_URL}/train/local",
                    json=payload,
                    timeout=1.5,
                )
                return
            except Exception as e:
                # Fall back to in-process background thread if service is down
                log.warning("LTS call failed (%s); using in-process thread.", e)

        # Fallback: run a background thread in this process (non-blocking)
        _start_background_training(cache, year, make, model_name, data_version or DATA_VERSION)

    except Exception as e:
        # Make sure the API path never breaks due to training side-effects
        log.error(f"[LOCAL {year}-{make}-{model_name}] training trigger failed: {e}")
        # Release lock on error
        release_training_lock(cache, year, make, model_name)


def _start_background_training(cache, year: int, make: str, model_name: str,
                               data_version: str) -> None:
    """Start background training thread"""
    # Capture the app instance from the current context
    from flask import current_app
    app = current_app._get_current_object()

    def _training_job():
        # Use the captured app instance for context
        with app.app_context():
            # Import kept inside to avoid circular imports and ensure app context is ready.
            try:
                log.info(f"[LOCAL {year}-{make}-{model_name}] background training thread started")
                from app.services.training_service import TrainingService
                result = TrainingService.train_local_from_db(year, make, model_name, data_version)
                if result:
                    log.info(f"[LOCAL {year}-{make}-{model_name}] training completed successfully")
                else:
                    log.warning(f"[LOCAL {year}-{make}-{model_name}] training returned None (insufficient data?)")
            except Exception:
                log.exception("[LOCAL %s-%s-%s] async training failed", year, make, model_name)
            finally:
                release_training_lock(cache, year, make, model_name)
                log.info(f"[LOCAL {year}-{make}-{model_name}] training lock released")

    log.info(f"[LOCAL {year}-{make}-{model_name}] starting background thread...")
    t = threading.Thread(target=_training_job, daemon=True)
    t.start()
    log.info(f"[LOCAL {year}-{make}-{model_name}] background thread created")
