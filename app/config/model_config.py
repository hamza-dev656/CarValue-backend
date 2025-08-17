# config/model_config.py
"""Environment-based configuration for model training and management."""

import os

# ===============================
# Redis Cache Configuration
# ===============================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")  # None if not set
CACHE_REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ===============================
# Model Training Configuration
# ===============================

# Hit-based local training
HITS_TTL_SEC = int(os.getenv("LOCAL_HITS_TTL_SEC", "86400"))        # 24h rolling window
TRAIN_LOCK_TTL = int(os.getenv("LOCAL_TRAIN_LOCK_TTL", "900"))      # 15m lock to prevent duplicates
HITS_THRESHOLD = int(os.getenv("LOCAL_TRAIN_HITS", "20"))           # trigger after N requests
LOCAL_TRAINER_URL = os.getenv("LOCAL_TRAINER_URL")                  # e.g. http://local-trainer:8080
DATA_VERSION = os.getenv("CARVALUE_DATA_VERSION", "v00000000")      # tag artifacts

# Model lifecycle management
LOCAL_MODEL_TTL_SEC = int(os.getenv("LOCAL_MODEL_TTL_SEC", "604800"))  # 7 days before local models expire
GLOBAL_MODEL_TTL_SEC = int(os.getenv("GLOBAL_MODEL_TTL_SEC", "2592000"))  # 30 days before global models expire

# Artifact directories
ARTIFACT_DIR = os.environ.get("CARVALUE_ARTIFACT_DIR", "./artifacts")
DATA_DIR = os.environ.get("CARVALUE_DATA_DIR", "./artifacts/data")

# Ensure directories exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# CatBoost Training Configuration
# ===============================

CB_TRAIN_DIR = os.getenv("CATBOOST_TRAIN_DIR", "/tmp/cbtrain")
CB_RAM_LIMIT = os.getenv("CATBOOST_RAM_LIMIT", "8gb")
CB_THREADS = int(os.getenv("CATBOOST_THREADS", "12"))
CB_ITERS = int(os.getenv("CB_ITERS", "900"))
CB_LR = float(os.getenv("CB_LR", "0.04"))
CB_DEPTH = int(os.getenv("CB_DEPTH", "7"))
CB_L2 = int(os.getenv("CB_L2", "10"))
CB_SUBSAMPLE = float(os.getenv("CB_SUBSAMPLE", "0.85"))
CB_COLSAMPLE = float(os.getenv("CB_COLSAMPLE", "0.85"))
CB_BORDER = int(os.getenv("CB_BORDER", "64"))
EARLY_STOP = int(os.getenv("CB_EARLY", "120"))

# ===============================
# Data Export Configuration
# ===============================

# Export/reuse toggles
REUSE_TSV = os.getenv("CARVALUE_REUSE_TSV", "1").lower() not in ("0", "false", "no")
FORCE_REEXPORT = os.getenv("CARVALUE_FORCE_REEXPORT", "0").lower() in ("1", "true", "yes")

# ===============================
# Local Model Configuration
# ===============================

# Local model features (simplified subset for local training)
LOCAL_FEATURE_COLS = ['year', 'listing_mileage', 'trim', 'dealer_state', 'exterior_color', 'style', 'driven_wheels', 'fuel_type', 'interior_color']
LOCAL_CATEGORICAL_COLS = ['trim', 'dealer_state', 'exterior_color', 'style', 'driven_wheels', 'fuel_type', 'interior_color']
