from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from app.repositories.vehicle_repository import VehicleRepository
from typing import List, Optional, Tuple


class RegressionService:
    @staticmethod
    def train_model_for_vehicle(
        year: int, make: str, model: str, features_to_use: List[str]
    ) -> Tuple[Optional[LinearRegression], Optional[StandardScaler], List[str]]:
        """
        Train a regression model using specified features
        Returns: (model, scaler, successfully_used_features)
        """
        filters = {
            "year": year,
            "make": make,
            "model": model,
            "listing_price": True,
            "listing_mileage": True,
        }
        listings = VehicleRepository.get_listings_with_filters(filters, limit=None)

        if not listings or len(listings) < 10:
            return None, None, []

        data = []
        for vehicle in listings:
            entry = {"price": vehicle.listing_price}

            if "mileage" in features_to_use:
                entry["mileage"] = vehicle.listing_mileage
            if "trim" in features_to_use:
                entry["trim"] = RegressionService._trim_to_numeric(vehicle.trim)
            if "color" in features_to_use:
                entry["color"] = RegressionService._is_premium_color(
                    vehicle.exterior_color
                )
            if "state" in features_to_use:
                entry["state"] = RegressionService._is_high_cost_state(
                    vehicle.dealer_state
                )

            data.append(entry)

        df = pd.DataFrame(data).dropna()

        available_features = [
            f for f in features_to_use if f in df.columns and len(df[f].unique()) > 1
        ]

        if len(available_features) == 0 or len(df) < 5:
            return None, None, [], None

        X = df[available_features]
        y = df["price"]

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            rmse = RegressionService._calculate_rmse(model, X_scaled, y)

            return model, scaler, available_features, rmse

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None, None, [], None

    @staticmethod
    def _calculate_rmse(
        model: LinearRegression,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Optional[float]:
        """
        Calculate RMSE directly from model predictions on training data
        """
        try:
            y_pred = model.predict(X)
            
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            return rmse
            
        except Exception as e:
            print(f"Error calculating RMSE: {str(e)}")
            return None

    @staticmethod
    def _trim_to_numeric(trim):
        """Convert trim level to a numeric value"""
        if not trim:
            return 0
        trim = trim.lower()

        premium_trims = ["limited", "platinum", "sport", "titanium", "premium"]
        budget_trims = ["base", "s", "lx", "le"]

        if any(p in trim for p in premium_trims):
            return 2
        elif any(b in trim for b in budget_trims):
            return 0
        return 1

    @staticmethod
    def _is_premium_color(color):
        """Check if color is considered premium"""
        if not color:
            return 0
        premium_colors = ["black", "white", "silver", "gray"]
        return 1 if color.lower() in premium_colors else 0

    @staticmethod
    def _is_high_cost_state(state):
        """Check if state is high cost"""
        if not state:
            return 0
        high_cost_states = ["CA", "NY", "WA"]
        return 1 if state.upper() in high_cost_states else 0
