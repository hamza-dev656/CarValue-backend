# app/services/vehicle_service.py
import os
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from app import cache
from app.repositories.vehicle_repository import VehicleRepository
from app.services.model_registry.model_registry import ModelRegistry
from app.services.bootstrap.data_processing import (
    prep_row_for_local_model, prep_row_for_catboost, 
    confidence_from_rmse_percentage, format_price
)
from app.utils.cache_utils import bump_hits, should_trigger_training
from app.services.training.training_utils import enqueue_local_training
from app.config.model_config import ARTIFACT_DIR

log = logging.getLogger(__name__)

# ---------- Service ----------
class VehicleService:
    """Business logic for vehicle valuation (CatBoost .cbm only)."""
    registry = ModelRegistry(cache)
    def calculate_market_value(year: int, make: str, model_name: str,
                           mileage: int = None, trim=None, color=None, dealer_state=None) -> Dict:
        
        # 1) Try local model first (if it exists)
        local_meta, local_model = VehicleService.registry.get_local_model(year, make, model_name)
        
        if local_meta and local_model:
            # build a request-level cache key
            ck = f"mv:local:{getattr(local_meta,'sha256','nohash')}:{year}:{make}:{model_name}:{mileage}:{trim}:{color}:{dealer_state}"
            cached = cache.get(ck)
            if cached:
                # Still bump hits for local model usage
                try:
                    bump_hits(cache, year, make, model_name)
                except Exception:
                    pass
                return cached

            # Prepare features for local model (simplified feature set)
            X = prep_row_for_local_model(year, mileage, trim, color, dealer_state)
            y_log = float(local_model.predict(X)[0])
            predicted_price = float(np.expm1(y_log))

            listings = VehicleRepository.get_listings_with_filters(
                {"year": year, "make": make, "model": model_name}
            )[:100]

            result = {
                "estimate": format_price(predicted_price),
                "listings": [v.to_dict() for v in listings],
                "calculation_date": datetime.utcnow().isoformat(),
                "method": "local_model",
                "model_accuracy": {
                    "rmse": None,
                    "rmse_percentage": round(getattr(local_meta, "rmse_pct", 0.0), 2),
                    "confidence": confidence_from_rmse_percentage(getattr(local_meta, "rmse_pct", None)),
                },
            }
            
            # Cache the result
            cache.set(ck, result, timeout=300)  # 5 minutes
            
            # Bump hit counter (for potential future retraining)
            try:
                bump_hits(cache, year, make, model_name)
            except Exception:
                pass
                
            return result
        
        # 2) Fall back to global model
        global_meta, global_model = VehicleService.registry.get_global_model()
        
        if global_meta and global_model:
            # build a request-level cache key (string only; no model objects)
            ck = f"mv:{getattr(global_meta,'sha256','nohash')}:{year}:{make}:{model_name}:{mileage}:{trim}:{color}:{dealer_state}"
            cached = cache.get(ck)
            if cached:
                return cached

            X = prep_row_for_catboost(year, make, model_name, mileage, trim, color, dealer_state)
            y_log = float(global_model.predict(X)[0])
            predicted_price = float(np.expm1(y_log))

            listings = VehicleRepository.get_listings_with_filters(
                {"year": year, "make": make, "model": model_name}
            )[:100]

            result = {
                "estimate": format_price(predicted_price),
                "listings": [v.to_dict() for v in listings],
                "calculation_date": datetime.utcnow().isoformat(),
                "method": "global_model",
                "model_accuracy": {
                    "rmse": None,
                    "rmse_percentage": round(getattr(global_meta, "rmse_pct", 0.0), 2),
                    "confidence": confidence_from_rmse_percentage(getattr(global_meta, "rmse_pct", None)),
                },
                "model_meta": {
                    "data_version": getattr(global_meta, "data_version", None),
                    "artifact": (global_meta.path or "").split("/")[-1],
                    "kind": getattr(global_meta, "kind", None),
                },
            }
            cache.set(ck, result, timeout=600)
            
            # Bump hit counter and maybe trigger local training
            try:
                count = bump_hits(cache, year, make, model_name)
                if should_trigger_training(count):
                    log.info(f"[TRIGGER] Hit threshold reached for {year}-{make}-{model_name}: {count} hits")
                    enqueue_local_training(cache, year, make, model_name)
            except Exception as e:
                log.error(f"[TRIGGER] Error in hit counter logic: {e}")
                pass
            
            return result

        # fallback if no model present
        result = VehicleService._get_fallback_estimate(year, make, model_name, mileage, trim, color, dealer_state)
        
        # Bump hit counter for fallback usage too (helps prioritize training)
        try:
            count = bump_hits(cache, year, make, model_name)
            if should_trigger_training(count):
                log.info(f"[TRIGGER] Hit threshold reached for {year}-{make}-{model_name}: {count} hits (fallback)")
                enqueue_local_training(cache, year, make, model_name)
        except Exception as e:
            log.error(f"[TRIGGER] Error in fallback hit counter logic: {e}")
            pass
            
        return result

    @staticmethod
    def _get_fallback_estimate(
        year: int,
        make: str,
        model: str,
        mileage: Optional[int] = None,
        trim: Optional[str] = None,
        color: Optional[str] = None,
        dealer_state: Optional[str] = None,
    ) -> Dict:
        filters = {"year": year, "make": make, "model": model, "listing_price": True}
        listings = VehicleRepository.get_listings_with_filters(filters)
        if not listings:
            return {
                "estimate": "N/A",
                "listings": [],
                "calculation_date": datetime.utcnow().isoformat(),
                "method": "no_data",
                "model_accuracy": {"confidence": "low", "rmse_percentage": None},
            }
        prices = [float(v.listing_price) for v in listings if v.listing_price is not None]
        avg_price = sum(prices) / max(len(prices), 1)
        return {
            "estimate": format_price(avg_price),
            "listings": [v.to_dict() for v in listings[:100]],
            "calculation_date": datetime.utcnow().isoformat(),
            "method": "simple_average",
            "model_accuracy": {"confidence": "low", "rmse_percentage": None},
        }
