# app/services/training_service.py
import os, csv, time, logging
from datetime import datetime
from typing import Optional, Iterable, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from app.repositories.vehicle_repository import VehicleRepository
from app.services.model_registry.model_registry import ModelRegistry, ModelMeta
from app import cache
from app.utils.constants import TRAIN_COLS, CAT_COL_INDEXES
from app.services.bootstrap.data_processing import state_to_region, to_float
from app.services.training.model_utils import (
    create_catboost_regressor, create_local_catboost_regressor, 
    write_catboost_column_description, calculate_rmse_percentage
)
from scripts.data_export import prepare_training_data
from app.config.model_config import EARLY_STOP

# -------- Logging --------
log = logging.getLogger("training")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s training: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)



class TrainingService:
    registry = ModelRegistry(cache)

    @staticmethod
    def train_global(data_version: str, limit_per_mm: int = 15000) -> Optional[ModelMeta]:
        """
        Train global CatBoost model using prepared training data.
        Data preparation is handled by scripts.data_export module.
        """
        log.info(f"[GLOBAL] Starting global model training (data_version={data_version}, limit_per_mm={limit_per_mm})")

        # Prepare training data (handles TSV export, reuse logic, column descriptions)
        train_path, valid_path, cdesc_path, n_train, n_valid = prepare_training_data(limit_per_mm=limit_per_mm)

        # Create CatBoost data pools
        log.info(f"[GLOBAL] Creating CatBoost data pools...")
        train_pool = Pool(data=train_path, column_description=cdesc_path, has_header=True, delimiter="\t")
        valid_pool = Pool(data=valid_path, column_description=cdesc_path, has_header=True, delimiter="\t")

        # Train model
        model = create_catboost_regressor()
        t1 = time.time()
        log.info(f"[GLOBAL] Training CatBoost model on {n_train:,} training samples...")
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=EARLY_STOP)
        log.info(f"[GLOBAL] Training completed in {time.time()-t1:.1f}s | best_iter={model.get_best_iteration()}")

        # Evaluate model on validation set
        log.info(f"[GLOBAL] Evaluating model on {n_valid:,} validation samples...")
        y_log_pred = model.predict(valid_pool)
        y_pred = np.expm1(y_log_pred.astype(np.float64))
        
        # Read true labels (log_price) from validation TSV
        y_true_price = []
        with open(valid_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)  # Skip header
            for row in r:
                y_true_price.append(float(np.expm1(float(row[0]))))
        
        y_true = np.array(y_true_price, dtype=np.float64)
        rmse_pct = calculate_rmse_percentage(y_pred, y_true)
        log.info(f"[GLOBAL] Model evaluation: RMSE% = {rmse_pct:.2f} (n_val={len(y_true):,})")

        # Save trained model
        from time import time as _now
        fname = f"global_{data_version}.cbm"
        path, sha256, size = TrainingService.registry.save_catboost(model, fname)
        
        # Create model metadata
        meta = ModelMeta(
            kind="global",
            segment=None,
            data_version=data_version,
            n_train=n_train,
            rmse_pct=float(rmse_pct),
            path=path,
            created_at=_now(),
            model_type="catboost_cbm",
            sha256=sha256,
            size_bytes=size,
        )
        
        # Register model in cache
        TrainingService.registry.register_model(meta)
        log.info(f"[GLOBAL] Model saved: {path} ({size/1e6:.1f}MB) sha256={sha256[:12]}...")
        log.info(f"[GLOBAL] Global model training completed successfully")
        return meta

    @staticmethod
    def train_local_from_db(year: int, make: str, model_name: str, data_version: str) -> Optional[ModelMeta]:
        """
        Train a local model for specific year/make/model from database data.
        Called by background thread when hit threshold is reached.
        """
        log.info(f"[LOCAL {year}-{make}-{model_name}] starting training (data_version={data_version})")
        
        try:
            from app.models.vehicle import Vehicle
            from app import db
            
            # Get data for this specific year/make/model
            vehicles = db.session.query(Vehicle).filter(
                Vehicle.year == year,
                Vehicle.make == make,
                Vehicle.model == model_name,
                Vehicle.listing_price.isnot(None),
                Vehicle.listing_mileage.isnot(None)
            ).all()
            
            if len(vehicles) < 50:
                log.warning(f"[LOCAL {year}-{make}-{model_name}] insufficient data ({len(vehicles)} records)")
                return None
            
            log.info(f"[LOCAL {year}-{make}-{model_name}] found {len(vehicles)} records")
            
            # Convert to DataFrame
            rows = []
            for v in vehicles:
                if v.listing_price and v.year:
                    rows.append({
                        "listing_price": float(v.listing_price),
                        "year": v.year,
                        "make": v.make,
                        "model": v.model,
                        "trim": v.trim,
                        "listing_mileage": v.listing_mileage or 0,
                        "dealer_state": v.dealer_state,
                        "exterior_color": v.exterior_color,
                        "style": v.style,
                        "driven_wheels": v.driven_wheels,
                        "fuel_type": v.fuel_type,
                        "interior_color": v.interior_color,
                        "used": getattr(v, "used", None),
                        "certified": getattr(v, "certified", None),
                        "listing_status": getattr(v, "listing_status", None),
                        "last_seen_date": getattr(v, "last_seen_date", None),
                    })
            
            if len(rows) < 50:
                log.warning(f"[LOCAL {year}-{make}-{model_name}] insufficient clean data ({len(rows)} records)")
                return None
                
            df = pd.DataFrame(rows)
            
            # Process features - simplified approach for local training
            # Add log_price target
            df['log_price'] = np.log1p(df['listing_price'])
            
            # Simple train/test split
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            if len(train_df) < 30:
                log.warning(f"[LOCAL {year}-{make}-{model_name}] insufficient training data ({len(train_df)} records)")
                return None
            
            log.info(f"[LOCAL {year}-{make}-{model_name}] training CatBoost with {len(train_df)} train, {len(test_df)} test records...")
            
            # Define features (similar to global training columns)
            feature_cols = ['year', 'listing_mileage']  # Start with numeric features
            cat_features = []  # Will be categorical column indices
            
            # Add categorical features if they have reasonable cardinality
            for col in ['trim', 'dealer_state', 'exterior_color', 'style', 'driven_wheels', 'fuel_type', 'interior_color']:
                if col in train_df.columns:
                    # Fill NaN and clean up
                    train_df[col] = train_df[col].fillna('Unknown').astype(str)
                    test_df[col] = test_df[col].fillna('Unknown').astype(str)
                    if train_df[col].nunique() < 50:  # Reasonable cardinality
                        feature_cols.append(col)
                        cat_features.append(len(feature_cols) - 1)  # Track categorical indices
            
            # Prepare training data
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df['log_price']
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df['log_price']
            
            # Train CatBoost model directly
            model = create_local_catboost_regressor(cat_features=cat_features)
            
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_true_price = np.expm1(y_test.astype(np.float64))
            y_pred_price = np.expm1(y_pred.astype(np.float64))
            
            rmse_pct = calculate_rmse_percentage(y_pred_price, y_true_price)
            
            log.info(f"[LOCAL {year}-{make}-{model_name}] RMSE%={rmse_pct:.2f} (n_train={len(train_df)}, n_test={len(test_df)})")
            
            # Save model
            fname = f"local_{year}_{make}_{model_name}_{data_version}.cbm"
            path, sha256, size = TrainingService.registry.save_catboost(model, fname)
            
            # Create meta
            from time import time as _now
            meta = ModelMeta(
                kind="local",
                segment=f"{year}:{make}:{model_name}",
                data_version=data_version,
                n_train=len(train_df),
                rmse_pct=float(rmse_pct),
                path=path,
                created_at=_now(),
                model_type="catboost_cbm",
                sha256=sha256,
                size_bytes=size,
            )
            
            # Register model
            TrainingService.registry.register_model(meta)
            
            log.info(f"[LOCAL {year}-{make}-{model_name}] saved {path} ({size/1e6:.1f}MB) RMSE%={rmse_pct:.2f}")
            return meta
            
        except Exception as e:
            log.exception(f"[LOCAL {year}-{make}-{model_name}] training failed: {e}")
            return None
