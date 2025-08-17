# app/utils/constants.py
"""
Static constants and mappings used across the application.
For environment-based configuration, see config.py
"""

# ===============================
# Regional Mapping
# ===============================

REGION_MAPPING = {
    # WEST
    "WA": "WEST", "OR": "WEST", "CA": "WEST", "NV": "WEST", "AZ": "WEST", "UT": "WEST", "ID": "WEST",
    "CO": "WEST", "NM": "WEST", "MT": "WEST", "WY": "WEST", "AK": "WEST", "HI": "WEST",
    # NORTHEAST
    "NY": "NORTHEAST", "NJ": "NORTHEAST", "PA": "NORTHEAST", "MA": "NORTHEAST", "CT": "NORTHEAST",
    "RI": "NORTHEAST", "NH": "NORTHEAST", "VT": "NORTHEAST", "ME": "NORTHEAST",
    # MIDWEST
    "IL": "MIDWEST", "OH": "MIDWEST", "MI": "MIDWEST", "IN": "MIDWEST", "WI": "MIDWEST", "MN": "MIDWEST",
    "IA": "MIDWEST", "MO": "MIDWEST", "KS": "MIDWEST", "NE": "MIDWEST", "SD": "MIDWEST", "ND": "MIDWEST",
    # SOUTH
    "TX": "SOUTH", "FL": "SOUTH", "GA": "SOUTH", "NC": "SOUTH", "SC": "SOUTH", "VA": "SOUTH", "MD": "SOUTH",
    "DC": "SOUTH", "DE": "SOUTH", "AL": "SOUTH", "MS": "SOUTH", "TN": "SOUTH", "KY": "SOUTH",
    "LA": "SOUTH", "AR": "SOUTH", "OK": "SOUTH", "WV": "SOUTH"
}

# ===============================
# Training Data Schema
# ===============================

# Training columns (Label = log_price first)
TRAIN_COLS = [
    "log_price", "year", "make", "model", "trim", "listing_mileage",
    "dealer_state", "exterior_color", "style", "driven_wheels", "fuel_type",
    "interior_color", "used", "certified", "listing_status", "region", "last_seen_date"
]

# Prediction columns (same order as training, but without log_price)
PREDICT_COLS = [
    "year", "make", "model", "trim", "listing_mileage",
    "dealer_state", "exterior_color", "style", "driven_wheels", "fuel_type",
    "interior_color", "used", "certified", "listing_status", "region", "last_seen_date"
]

# Categorical column indexes for CatBoost column_description (0 is Label)
CAT_COL_INDEXES = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # make, model, trim, state, ...

