# app/services/vehicle_service.py

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app import cache
from app.repositories.vehicle_repository import VehicleRepository

# --- State 1 dependencies (Regression) ---
from app.services.regression_service import RegressionService

# --- State 2 dependencies (CatBoost/Registry) ---
from app.services.model_registry.model_registry import ModelRegistry
from app.services.bootstrap.data_processing import (
    prep_row_for_local_model,
    prep_row_for_catboost,
    confidence_from_rmse_percentage,
    format_price as cb_format_price,
)
from app.utils.cache_utils import bump_hits, should_trigger_training
from app.services.training.training_utils import enqueue_local_training

log = logging.getLogger(__name__)

# ============================================================
# ===============  STATE 1: REGRESSION VERSION  ==============
# ============================================================

class VehicleServiceRegression:
    """Business logic for vehicle valuation (Regression-based)"""

    @staticmethod
    @cache.memoize(timeout=3600)
    def calculate_market_value(
        year: int,
        make: str,
        model: str,
        mileage: int = None,
        trim: Optional[str] = None,
        color: Optional[str] = None,
        dealer_state: Optional[str] = None,
    ) -> Dict:
        """
        Calculate market value with RMSE scores and confidence levels
        """
        requested_features = []
        if mileage is not None:
            requested_features.append("mileage")
        if trim is not None:
            requested_features.append("trim")
        if color is not None:
            requested_features.append("color")
        if dealer_state is not None:
            requested_features.append("state")

        model_key = f"regression_{year}_{make}_{model}_" + "_".join(
            sorted(requested_features)
        )
        cached_result = cache.get(model_key)
        regression_model, scaler, used_features, rmse = (
            cached_result if cached_result else (None, None, [], None)
        )

        if not regression_model and requested_features:
            regression_model, scaler, used_features, rmse = (
                RegressionService.train_model_for_vehicle(
                    year, make, model, requested_features
                )
            )
            if regression_model:
                cache.set(model_key, (regression_model, scaler, used_features, rmse))

        if regression_model and scaler and used_features:
            input_features = {}
            if "mileage" in used_features:
                input_features["mileage"] = mileage
            if "trim" in used_features:
                input_features["trim"] = RegressionService._trim_to_numeric(trim)
            if "color" in used_features:
                input_features["color"] = RegressionService._is_premium_color(color)
            if "state" in used_features:
                input_features["state"] = RegressionService._is_high_cost_state(
                    dealer_state
                )

            input_df = pd.DataFrame([input_features])
            input_scaled = scaler.transform(input_df)

            predicted_price = regression_model.predict(input_scaled)[0]

            listings = VehicleRepository.get_listings_with_filters(
                {"year": year, "make": make, "model": model}
            )

            rmse_percentage = None
            confidence = "medium"

            if rmse is not None and predicted_price > 0:
                rmse_percentage = (rmse / predicted_price) * 100
                confidence = VehicleServiceRegression._determine_confidence(rmse_percentage)
            elif rmse is None and len(listings) >= 10:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "estimate": VehicleServiceRegression._format_price(predicted_price),
                "listings": [vehicle.to_dict() for vehicle in listings],
                "calculation_date": datetime.utcnow().isoformat(),
                "method": f"regression ({', '.join(used_features)})",
                "model_accuracy": {
                    "rmse": round(rmse, 2) if rmse is not None else None,
                    "rmse_percentage": (
                        round(rmse_percentage, 2)
                        if rmse_percentage is not None
                        else None
                    ),
                    "confidence": confidence,
                },
                "training_metrics": {
                    "features_used": used_features,
                    "samples_used": len(listings),
                },
            }

        return VehicleServiceRegression._get_fallback_estimate(
            year, make, model, mileage, trim, color, dealer_state
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
        """Improved fallback method with accuracy reporting"""
        filters = {
            "year": year,
            "make": make,
            "model": model,
            "listing_price": True,
        }

        if mileage:
            filters["mileage"] = mileage
        if trim:
            filters["trim"] = trim
        if color:
            filters["color"] = color
        if dealer_state:
            filters["dealer_state"] = dealer_state

        listings = VehicleRepository.get_listings_with_filters(filters)
        avg_price = VehicleRepository.get_average_price_with_filters(filters)

        if avg_price:
            original_price = avg_price
            if mileage:
                avg_price = VehicleServiceRegression._adjust_for_mileage(avg_price, mileage, year)
            if trim:
                avg_price = VehicleServiceRegression._adjust_for_trim(avg_price, trim)
            if color:
                avg_price = VehicleServiceRegression._adjust_for_body_color(avg_price, color)
            if dealer_state:
                avg_price = VehicleServiceRegression._adjust_for_state(avg_price, dealer_state)

            price_diff = abs(avg_price - original_price)
            error_pct = (
                (price_diff / original_price) * 100 if original_price > 0 else None
            )

            return {
                "estimate": VehicleServiceRegression._format_price(avg_price),
                "listings": [vehicle.to_dict() for vehicle in listings],
                "calculation_date": datetime.utcnow().isoformat(),
                "method": "adjusted_average",
                "model_accuracy": {
                    "confidence": VehicleServiceRegression._determine_fallback_confidence(
                        len(listings), error_pct
                    ),
                    "rmse": None,
                    "rmse_percentage": (
                        round(error_pct, 2) if error_pct is not None else None
                    ),
                },
            }

        return {
            "estimate": "N/A",
            "listings": [],
            "calculation_date": datetime.utcnow().isoformat(),
            "method": "no_data",
            "model_accuracy": {
                "confidence": "low",
                "rmse": None,
                "rmse_percentage": None,
            },
        }

    @staticmethod
    def _determine_confidence(rmse_percentage: Optional[float]) -> str:
        """Determine confidence level based on RMSE percentage"""
        if rmse_percentage is None:
            return "medium"

        if rmse_percentage < 5:
            return "high"
        elif rmse_percentage < 15:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _determine_fallback_confidence(
        sample_size: int, error_pct: Optional[float]
    ) -> str:
        """Determine confidence for fallback method"""
        if sample_size == 0:
            return "low"

        if error_pct is None:
            return "medium" if sample_size >= 5 else "low"

        if error_pct < 10 and sample_size >= 10:
            return "medium"
        elif error_pct < 20 and sample_size >= 5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _format_price(price: float) -> str:
        """Format price as currency rounded to nearest $100"""
        if price is None:
            return "N/A"
        rounded = round(price / 100) * 100
        return f"${rounded:,.0f}"

    @staticmethod
    def get_makes_and_models() -> Dict[int, Dict[str, List[str]]]:
        """Get all available years with makes and models"""
        result = {}
        current_year = datetime.now().year

        years = range(1990, current_year + 1)

        for year in years:
            makes = VehicleRepository.get_makes_by_year(year)
            make_models = {}

            for make in makes:
                models = VehicleRepository.get_models_by_make_year(year, make)
                make_models[make] = models

            if make_models:
                result[year] = make_models

        return result

    @staticmethod
    def _adjust_for_mileage(base_price: float, mileage: int, year: int) -> float:
        """Apply non-linear mileage adjustment"""
        current_year = datetime.now().year
        vehicle_age = current_year - year

        base_depreciation = 0.10 

        avg_miles_per_year = 12000
        expected_miles = vehicle_age * avg_miles_per_year
        mileage_diff = mileage - expected_miles

        if mileage_diff > 0:
            adjustment_factor = 0.15 + (mileage_diff / 100000 * 0.05)
            adjustment = mileage_diff * adjustment_factor
        else:
            adjustment = mileage_diff * 0.08

        adjusted_price = base_price - adjustment

        return max(adjusted_price, base_price * 0.3)

    @staticmethod
    def _adjust_for_trim(price: float, trim: str) -> float:
        trim = trim.lower()
        premium_trims = [
            "limited",
            "platinum",
            "sport",
            "titanium",
            "premium",
            "xlt",
            "lariat",
        ]
        budget_trims = ["base", "s", "lx", "le"]

        if any(keyword in trim for keyword in premium_trims):
            price *= 1.10
        elif any(keyword in trim for keyword in budget_trims):
            price *= 0.95
        return price

    @staticmethod
    def _adjust_for_body_color(price: float, body_color: str) -> float:
        body_color = body_color.lower()
        if body_color in ["black", "white", "silver", "gray"]:
            return price * 1.08
        elif body_color in ["wind chill pearl", "ice cap", "red"]:
            return price * 0.97
        return price

    @staticmethod
    def _adjust_for_state(price: float, state: str) -> float:
        state = state.upper()
        high_cost_states = {"CA": 1.20, "NY": 1.08, "WA": 1.07}
        low_cost_states = {"TX": 0.97, "OH": 0.95, "MI": 0.93}

        if state in high_cost_states:
            return price * high_cost_states[state]
        elif state in low_cost_states:
            return price * low_cost_states[state]
        return price

# ============================================================
# ==============  STATE 2: CATBOOST/REGISTRY VERSION  ========
# ============================================================

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


class VehicleServiceCatBoost:
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
       
        local_meta, local_model = VehicleServiceCatBoost.registry.get_local_model(year, make, model_name)
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
                    "estimate": cb_format_price(predicted_price),
                    "listings": [v.to_dict() for v in listings],
                    "calculation_date": datetime.now(timezone.utc).isoformat(),
                    "method": "local_model",
                    "model_accuracy": {
                        "rmse": None,
                        "rmse_percentage": round(getattr(local_meta, "rmse_pct", 0.0), 2),
                        "confidence": confidence_from_rmse_percentage(getattr(local_meta, "rmse_pct", None)),
                    },
                    "model_meta": {
                        "data_version": getattr(local_meta, "data_version", None),
                        "artifact": (local_meta.path or "").split("/")[-1],
                        "kind": getattr(local_meta, "kind", None),
                    }
                }
                cache.set(ck, result, timeout=300)  # 5 minutes
                return result

            except Exception as e:
                # If local predict fails, log and continue to GLOBAL (do not bump hits here;
                log.exception(f"[LOCAL {year}-{make}-{model_name}] prediction failed, falling back to global: {e}")

        #  Try GLOBAL model
        global_meta, global_model = VehicleServiceCatBoost.registry.get_global_model()

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
                    "estimate": cb_format_price(predicted_price),
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
        return VehicleServiceCatBoost._get_fallback_estimate(
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
            "estimate": cb_format_price(avg_price),
            "listings": [v.to_dict() for v in listings[:100]],
            "calculation_date": datetime.now(timezone.utc).isoformat(),
            "method": "simple_average",
            "model_accuracy": {"confidence": "low", "rmse_percentage": None},
        }

# ============================================================
# ===================  ENV-DRIVEN FACADE  ====================
# ============================================================

_BACKEND = os.getenv("VEHICLE_MV_BACKEND", "catboost").strip().lower()
# Allowed: "regression", "catboost"

class VehicleService:
    """
    Facade that dispatches to the selected backend based on the VEHICLE_MV_BACKEND env var.
      - "regression" -> VehicleServiceRegression (State 1)
      - "catboost"   -> VehicleServiceCatBoost (State 2)
    """

    @staticmethod
    def calculate_market_value(
        year: int,
        make: str,
        model: str,
        mileage: int = None,
        trim: Optional[str] = None,
        color: Optional[str] = None,
        dealer_state: Optional[str] = None,
    ) -> Dict:
        if _BACKEND == "regression":
            # State 1 expects "model" param to be called "model"
            return VehicleServiceRegression.calculate_market_value(
                year, make, model, mileage, trim, color, dealer_state
            )
        # State 2 expects "model_name" but the signature is identical positionally
        return VehicleServiceCatBoost.calculate_market_value(
            year, make, model, mileage, trim, color, dealer_state
        )

    # Optional pass-throughs (only implemented in regression state originally).
    # Expose them here for convenience—if you need them and backend is catboost,
    # you can still call the regression helpers as utility functions.

    @staticmethod
    def get_makes_and_models() -> Dict[int, Dict[str, List[str]]]:
        return VehicleServiceRegression.get_makes_and_models()

    # Keep fallbacks/formatters accessible if needed by callers/tests.
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
        if _BACKEND == "regression":
            return VehicleServiceRegression._get_fallback_estimate(
                year, make, model, mileage, trim, color, dealer_state
            )
        return VehicleServiceCatBoost._get_fallback_estimate(
            year, make, model, mileage, trim, color, dealer_state
        )
