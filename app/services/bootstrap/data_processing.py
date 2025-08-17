# app/utils/data_processing.py
"""
Common data processing utilities used across services
"""

import numpy as np
import pandas as pd
from typing import Optional
from app.utils.constants import REGION_MAPPING, PREDICT_COLS


def norm_str(x) -> str:
    """Normalize string value, return 'UNK' for null/empty values"""
    return "UNK" if x in (None, "", "None", "nan") else str(x)


def norm_miles(x) -> float:
    """Normalize mileage value, return 0.0 for invalid values"""
    try:
        return float(x)
    except Exception:
        return 0.0


def to_float(x) -> Optional[float]:
    """Convert value to float, return None for invalid values"""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def state_to_region(state: Optional[str]) -> str:
    """Map US state code to geographic region."""
    return REGION_MAPPING.get((state or "").upper(), "UNK")



def prep_row_for_local_model(year: int, mileage: Optional[float], trim: Optional[str], 
                            color: Optional[str], dealer_state: Optional[str]) -> pd.DataFrame:
    """Build one-row DataFrame with features matching local model training
    
    Based on actual trained model, expects: ['year', 'listing_mileage', 'style', 'driven_wheels', 'fuel_type']
    with categorical indices [2, 3, 4] meaning style, driven_wheels, fuel_type are categorical
    """
    row = {
        "year": int(year),
        "listing_mileage": float(mileage) if mileage else 0.0,
        # Only include the exact features the model was trained with
        "style": "Unknown",          # categorical index 2
        "driven_wheels": "Unknown",  # categorical index 3  
        "fuel_type": "Unknown",      # categorical index 4
    }
    
    df = pd.DataFrame([row])
    
    # Ensure exact data types as training
    df['year'] = df['year'].astype('int64')
    df['listing_mileage'] = df['listing_mileage'].astype('float64')
    df['style'] = df['style'].astype(str)
    df['driven_wheels'] = df['driven_wheels'].astype(str)
    df['fuel_type'] = df['fuel_type'].astype(str)
    
    # Ensure exact column order as expected by model
    return df[['year', 'listing_mileage', 'style', 'driven_wheels', 'fuel_type']]


def prep_row_for_catboost(year: int, make: str, model: str, mileage: Optional[float], 
                         trim: Optional[str], color: Optional[str], 
                         dealer_state: Optional[str]) -> pd.DataFrame:
    """Build one-row DataFrame with the exact schema CatBoost was trained on."""
    row = {
        "year": int(year),
        "make": norm_str(make),
        "model": norm_str(model),
        "trim": norm_str(trim),
        "listing_mileage": norm_miles(mileage),
        "dealer_state": norm_str(dealer_state),
        "exterior_color": norm_str(color),
        # fields that aren't provided by the API → default to UNK
        "style": "UNK",
        "driven_wheels": "UNK",
        "fuel_type": "UNK",
        "interior_color": "UNK",
        # booleans trained as YES/NO strings
        "used": "YES",
        "certified": "NO",
        "listing_status": "UNK",
        # derived from state
        "region": state_to_region(dealer_state),
        # treated as Categorical in training; we can leave it empty
        "last_seen_date": "",
    }
    df = pd.DataFrame([row])[PREDICT_COLS]  # enforce column order
    # ensure categorical-like columns are strings
    for c in ["make", "model", "trim", "dealer_state", "exterior_color", "style", "driven_wheels",
              "fuel_type", "interior_color", "used", "certified", "listing_status", "region", "last_seen_date"]:
        df[c] = df[c].astype("string")
    # numeric types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(np.int32)
    df["listing_mileage"] = pd.to_numeric(df["listing_mileage"], errors="coerce").fillna(0.0).astype(np.float32)
    return df


def confidence_from_rmse_percentage(rmse_percentage: Optional[float]) -> str:
    """Convert RMSE percentage to confidence level"""
    if rmse_percentage is None:
        return "medium"
    if rmse_percentage < 5:
        return "high"
    if rmse_percentage < 15:
        return "medium"
    return "low"


def format_price(price: Optional[float]) -> str:
    """Format price as currency string"""
    if price is None:
        return "N/A"
    rounded = round(price / 100) * 100
    return f"${rounded:,.0f}"
