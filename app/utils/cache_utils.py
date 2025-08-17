# app/utils/cache_utils.py
"""
Cache utilities for hit counting and locking mechanisms
"""

import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

from app.config.model_config import HITS_TTL_SEC, TRAIN_LOCK_TTL, HITS_THRESHOLD


def hits_key(year: int, make: str, model_name: str) -> str:
    """Generate cache key for hit counter"""
    return f"hits:{year}:{make}:{model_name}"


def lock_key(year: int, make: str, model_name: str) -> str:
    """Generate cache key for training lock"""
    return f"lock:local:{year}:{make}:{model_name}"


def bump_hits(cache, year: int, make: str, model_name: str) -> int:
    """Increment per-YMM counter with TTL. Works on SimpleCache and RedisCache."""
    key = hits_key(year, make, model_name)
    try:
        n = cache.get(key)
        if n is None:
            # first hit in window
            cache.set(key, 1, timeout=HITS_TTL_SEC)
            log.info(f"[HIT COUNTER] {year}-{make}-{model_name}: 1 (first hit)")
            return 1
        # bump; NOTE: on SimpleCache this resets TTL (OK for our use)
        new_n = int(n) + 1
        cache.set(key, new_n, timeout=HITS_TTL_SEC)
        if new_n % 5 == 0 or new_n >= HITS_THRESHOLD:  # Log every 5th hit and at threshold
            log.info(f"[HIT COUNTER] {year}-{make}-{model_name}: {new_n} hits")
        return new_n
    except Exception as e:
        log.warning(f"[HIT COUNTER] Error bumping hits for {year}-{make}-{model_name}: {e}")
        return 1


def is_training_locked(cache, year: int, make: str, model_name: str) -> bool:
    """Check if training is already in progress for this YMM"""
    key = lock_key(year, make, model_name)
    return bool(cache.get(key))


def acquire_training_lock(cache, year: int, make: str, model_name: str) -> bool:
    """Acquire training lock. Returns True if lock was acquired, False if already locked"""
    key = lock_key(year, make, model_name)
    if cache.get(key):
        return False
    
    cache.set(key, "1", timeout=TRAIN_LOCK_TTL)
    return True


def release_training_lock(cache, year: int, make: str, model_name: str) -> None:
    """Release training lock"""
    key = lock_key(year, make, model_name)
    cache.delete(key)


def should_trigger_training(hit_count: int) -> bool:
    """Check if hit count has reached the threshold for triggering training"""
    return hit_count >= HITS_THRESHOLD
