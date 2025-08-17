# app/services/vehicle_service.py
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

from app import cache
from app.repositories.vehicle_repository import VehicleRepository
from app.services.model_registry.model_registry import ModelRegistry
from app.services.bootstrap.data_processing import (
    prep_row_for_local_model,
    prep_row_for_catboost,
    confidence_from_rmse_percentage,
    format_price,
)
from app.utils.cache_utils import bump_hits, should_trigger_training
from app.services.training.training_utils import enqueue_local_training

log = logging.getLogger(__name__)


def _cache_key(prefix: str, meta, year, make, model_name, mileage, trim, color, dealer_state) -> str:
    sha = getattr(meta, "sha256", "nohash") if meta else "nometa"
    return f"mv:{prefix}:{sha}:{year}:{make}:{model_name}:{mileage}:{trim}:{color}:{dealer_state}"


def _bump_and_maybe_enqueue(year: int, make: str, model_name: str) -> None:
    """Increment hits and enqueue local training if threshold reached."""
    try:
        count = bump_hits(cache, year, make, model_name)
        if should_trigger_training(count):
            log.info(f"[TRIGGER] Hit threshold reached for {year}-{make}-{model_name}: {count} hits")
            enqueue_local_training(cache, year, make, model_name)
    except Exception as e:
        log.error(f"[TRIGGER] Error in hit counter logic for {year}-{make}-{model_name}: {e}")


class VehicleService:
    """Business logic for vehicle valuation (CatBoost .cbm only)."""
    registry = ModelRegistry(cache)

    @staticmethod
    def calculate_market_value(
        year: int,
        make: str,
        model_name: str,
        mileage: int = None,
        trim: Optional[str] = None,
        color: Optional[str] = None,
        dealer_state: Optional[str] = None,
    ) -> Dict:
       
        local_meta, local_model = VehicleService.registry.get_local_model(year, make, model_name)
        if local_meta and local_model:
            ck = _cache_key("local", local_meta, year, make, model_name, mileage, trim, color, dealer_state)
            cached = cache.get(ck)
            if cached:
                return cached

            try:
                X = prep_row_for_local_model(year, mileage, trim, color, dealer_state)
                y_log = float(local_model.predict(X)[0])
                predicted_price = float(np.expm1(y_log))

                listings = VehicleRepository.get_listings_with_filters(
                    {"year": year, "make": make, "model": model_name}
                )[:100]

                result = {
                    "estimate": format_price(predicted_price),
                    "listings": [v.to_dict() for v in listings],
                    "calculation_date": datetime.now(timezone.utc).isoformat(),
                    "method": "local_model",
                    "model_accuracy": {
                        "rmse": None,
                        "rmse_percentage": round(getattr(local_meta, "rmse_pct", 0.0), 2),
                        "confidence": confidence_from_rmse_percentage(getattr(local_meta, "rmse_pct", None)),
                    },
                }
                cache.set(ck, result, timeout=300)  # 5 minutes
                return result

            except Exception as e:
                # If local predict fails, log and continue to GLOBAL (do not bump hits here;

                log.exception(f"[LOCAL {year}-{make}-{model_name}] prediction failed, falling back to global: {e}")

        #  Try GLOBAL model
        global_meta, global_model = VehicleService.registry.get_global_model()

        # Always bump hits when there is NO local model (cache hit or not)
        if not (local_meta and local_model):
            _bump_and_maybe_enqueue(year, make, model_name)

        if global_meta and global_model:
            ck = _cache_key("global", global_meta, year, make, model_name, mileage, trim, color, dealer_state)
            cached = cache.get(ck)
            if cached:
                return cached

            try:
                X = prep_row_for_catboost(year, make, model_name, mileage, trim, color, dealer_state)
                y_log = float(global_model.predict(X)[0])
                predicted_price = float(np.expm1(y_log))

                listings = VehicleRepository.get_listings_with_filters(
                    {"year": year, "make": make, "model": model_name}
                )[:100]

                result = {
                    "estimate": format_price(predicted_price),
                    "listings": [v.to_dict() for v in listings],
                    "calculation_date": datetime.now(timezone.utc).isoformat(),
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
                return result

            except Exception as e:
                # If GLOBAL predict fails, log and go to fallback
                log.exception(f"[GLOBAL] prediction failed for {year}-{make}-{model_name}; using fallback: {e}")

        # Fallback (simple average) 
        return VehicleService._get_fallback_estimate(
            year, make, model_name, mileage, trim, color, dealer_state
        )

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
                "calculation_date": datetime.now(timezone.utc).isoformat(),
                "method": "no_data",
                "model_accuracy": {"confidence": "low", "rmse_percentage": None},
            }

        prices = [float(v.listing_price) for v in listings if v.listing_price is not None]
        avg_price = sum(prices) / max(len(prices), 1)

        return {
            "estimate": format_price(avg_price),
            "listings": [v.to_dict() for v in listings[:100]],
            "calculation_date": datetime.now(timezone.utc).isoformat(),
            "method": "simple_average",
            "model_accuracy": {"confidence": "low", "rmse_percentage": None},
        }

    @staticmethod
    def get_makes_and_models() -> Dict:
        """Get available makes and models organized by year"""
        try:
            vehicles = VehicleRepository.get_all_vehicles()

            data = {}
            for vehicle in vehicles:
                if not vehicle.year or not vehicle.make or not vehicle.model:
                    continue

                year_str = str(vehicle.year)
                make = vehicle.make
                model = vehicle.model

                if year_str not in data:
                    data[year_str] = {}

                if make not in data[year_str]:
                    data[year_str][make] = set()

                data[year_str][make].add(model)

            # Convert sets to sorted lists
            for year in data:
                for mmake in data[year]:
                    data[year][mmake] = sorted(list(data[year][mmake]))

            return data

        except Exception as e:
            log.error(f"Error getting makes and models: {e}")
            return {}
