"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    # Months since January 2021, so the model can follow prices over time.
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    # One 0/1 column per property type.
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X


TREE_NUMERIC_FEATURES = [
    "square_feet", "area_is_range", "bedrooms", "washrooms", "parking_total",
    "garage", "garage_gt_parking", "lot_front", "lot_depth", "lot_area",
    "lot_missing", "latitude", "longitude", "months_since_2021",
]
TREE_CATEGORICAL_FEATURES = ["house_type_name", "house_style", "municipality", "community"]
TREE_RAW_FEATURES = [
    "house_area", "bedrooms", "washrooms", "parking_total", "garage", "lot_front",
    "lot_depth", "latitude", "longitude", "date_listed", *TREE_CATEGORICAL_FEATURES,
]


def tree_features(homes, track):
    """Deterministic whitelist; no targets, IDs, text or after-sale fields are read."""
    if track not in ("off_market", "list_aware"):
        raise ValueError(f"Unknown track: {track}")
    raw = TREE_RAW_FEATURES + (["list_price"] if track == "list_aware" else [])
    source = homes.loc[:, raw]
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(source["house_area"])
    X["area_is_range"] = source["house_area"].astype(str).str.contains("-", regex=False).astype(int)
    for column in ["bedrooms", "washrooms", "parking_total", "garage", "lot_front",
                   "lot_depth", "latitude", "longitude"]:
        X[column] = pd.to_numeric(source[column], errors="raise")
    X["garage_gt_parking"] = (X["garage"] > X["parking_total"]).astype(int)
    X["lot_area"] = X["lot_front"] * X["lot_depth"]
    X["lot_missing"] = X[["lot_front", "lot_depth"]].isna().any(axis=1).astype(int)
    X["months_since_2021"] = (source["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    for column in TREE_CATEGORICAL_FEATURES:
        X[column] = source[column]
    numeric = TREE_NUMERIC_FEATURES + (["list_price"] if track == "list_aware" else [])
    if track == "list_aware":
        X["list_price"] = pd.to_numeric(source["list_price"], errors="raise")
    return X[numeric + TREE_CATEGORICAL_FEATURES]
