# app/utils/model_utils.py
"""
Model utilities for CatBoost operations and evaluation
"""

import numpy as np
from catboost import CatBoostRegressor
from app.utils.constants import TRAIN_COLS, CAT_COL_INDEXES
from app.config.model_config import (
    CB_TRAIN_DIR, CB_RAM_LIMIT, CB_THREADS, CB_ITERS, CB_LR, 
    CB_DEPTH, CB_L2, CB_SUBSAMPLE, CB_COLSAMPLE, CB_BORDER, EARLY_STOP
)


def create_catboost_regressor(iterations: int = None, learning_rate: float = None, 
                             depth: int = None, verbose: bool = True) -> CatBoostRegressor:
    """Create a CatBoost regressor with standard configuration"""
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations or CB_ITERS,
        learning_rate=learning_rate or CB_LR,
        depth=depth or CB_DEPTH,
        l2_leaf_reg=CB_L2,
        subsample=CB_SUBSAMPLE,
        colsample_bylevel=CB_COLSAMPLE,
        used_ram_limit=CB_RAM_LIMIT,
        allow_writing_files=True,
        train_dir=CB_TRAIN_DIR,
        max_ctr_complexity=1,
        one_hot_max_size=16,
        border_count=CB_BORDER,
        od_type="Iter",
        random_seed=42,
        thread_count=CB_THREADS,
        verbose=200 if verbose else False,
    )


def create_local_catboost_regressor(cat_features=None) -> CatBoostRegressor:
    """Create a CatBoost regressor optimized for local training"""
    return CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
        cat_features=cat_features or [],
        allow_writing_files=False
    )


def write_catboost_column_description(path: str) -> None:
    """Write CatBoost column description file"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("0\tLabel\n")      # log_price
        f.write("1\tNum\n")        # year
        for idx in range(2, len(TRAIN_COLS)):
            kind = "Categ" if idx in CAT_COL_INDEXES else "Num"
            f.write(f"{idx}\t{kind}\n")


def calculate_rmse_percentage(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate RMSE as percentage of median true value"""
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    denom = float(max(np.median(y_true), 1.0))
    return (rmse / denom) * 100.0
