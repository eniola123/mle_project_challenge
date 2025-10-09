#!/usr/bin/env python3
"""
app.py

FastAPI app for Sound Realty House Price Prediction.

Endpoints:
- GET  /               : basic welcome
- GET  /health         : health check + model metadata
- POST /predict        : Accepts the columns from future_unseen_examples.csv (arbitrary fields),
                         requires 'zipcode' to join demographics server-side. Returns prediction + metadata.
- POST /predict_required : Accepts only the features required by the trained model
                           (exact names in model/model_features.json) and returns a prediction.
- POST /predict_sales  : Minimal sales-only endpoint: accepts sales subset + zipcode and returns prediction.

Expectations:
- model/model.pkl and model/model_features.json exist (created by train_model.py).
- data/zipcode_demographics.csv exists (used to enrich requests by zipcode).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

# ------------------------
# Configuration / paths
# ------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.pkl"
FEATURES_PATH = ROOT / "model" / "model_features.json"
DEMOGRAPHICS_PATH = ROOT / "data" / "zipcode_demographics.csv"

# Minimal sale features used by sales personnel (predict_sales)
SALE_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sound_realty_api")

# ------------------------
# Load model and metadata
# ------------------------
def load_model_and_features(model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list file not found at: {features_path}")

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    with open(features_path, "r") as f:
        features = json.load(f)

    return model_obj, features


def load_demographics(path: Path = DEMOGRAPHICS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Demographics CSV not found at: {path}")
    df = pd.read_csv(path, dtype={"zipcode": str})
    return df


# Try loading at startup (will raise if missing)
try:
    model, MODEL_FEATURES = load_model_and_features()
    logger.info(f"Loaded model from {MODEL_PATH} with {len(MODEL_FEATURES)} features.")
except Exception as e:
    # Do not crash the process here; re-raise so user is informed.
    logger.exception("Failed to load model/features at startup.")
    raise

try:
    DEMOGRAPHICS_DF = load_demographics()
    logger.info(f"Loaded demographics from {DEMOGRAPHICS_PATH} with {len(DEMOGRAPHICS_DF)} rows.")
except Exception as e:
    logger.exception("Failed to load demographics CSV at startup.")
    raise

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Sound Realty Price Predictor", version="1.0")


@app.get("/")
def root():
    return {"message": "Sound Realty Price Prediction API - ready"}


@app.get("/health")
def health():
    """Return basic health information and model metadata."""
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "n_features": len(MODEL_FEATURES),
        "features_sample": MODEL_FEATURES[:10],
    }


def _merge_demographics(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge single-row dataframe with demographics using 'zipcode'.
    - Ensures zipcode is a string.
    - Fills missing demographic columns using medians from demographics DF.
    - Drops zipcode column afterward.
    """
    if "zipcode" not in row_df.columns:
        raise ValueError("zipcode column is required to merge demographics.")

    # Ensure zipcode is string
    row_df["zipcode"] = row_df["zipcode"].astype(str)

    # Left-join to demographics
    merged = row_df.merge(DEMOGRAPHICS_DF, how="left", on="zipcode")

    # Fill demographic NaNs with column medians from DEMOGRAPHICS_DF
    dem_cols = [c for c in DEMOGRAPHICS_DF.columns if c != "zipcode"]
    for c in dem_cols:
        if c not in merged.columns:
            # If the demographics CSV has a column but merged doesn't, create and fill with median
            merged[c] = DEMOGRAPHICS_DF[c].median()
        else:
            # If the merge resulted in NaN, fill with median
            if merged[c].isnull().any():
                merged[c] = merged[c].fillna(DEMOGRAPHICS_DF[c].median())

    # Drop zipcode before feeding to model (the model doesn't use zipcode directly)
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])

    return merged


def _prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe contains all MODEL_FEATURES in the exact order expected by the model.
    Missing features are filled with 0.0 (or could be changed to median/imputation).
    Non-numeric entries are coerced to numeric where possible.
    """
    # Add missing features
    for f in MODEL_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    # Keep only model features in the proper order
    X = df[MODEL_FEATURES].copy()

    # Attempt numeric conversion where applicable
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    """
    Accept arbitrary JSON payload that contains the columns from future_unseen_examples.csv.
    Must include 'zipcode' so demographic features can be merged.
    Returns single prediction and metadata.
    """
    try:
        # Convert payload to single-row DataFrame
        row = pd.DataFrame([payload])

        if "zipcode" not in row.columns:
            raise HTTPException(status_code=400, detail="Field 'zipcode' is required in payload for demographic join.")

        # Merge demographics and prepare features
        merged = _merge_demographics(row)
        X = _prepare_model_input(merged)

        # Predict
        pred = model.predict(X)[0]
        return {
            "prediction": float(pred),
            "model": "sound_realty_knn",
            "n_features_used": len(MODEL_FEATURES),
            "model_features": MODEL_FEATURES,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /predict")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_required")
def predict_required(payload: Dict[str, Any] = Body(...)):
    """
    Accepts only the features required by the model (exact keys in model_features.json).
    Returns prediction.
    """
    try:
        # Validate presence of all required features
        missing = [f for f in MODEL_FEATURES if f not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

        # Build DataFrame in correct order
        row = pd.DataFrame([{f: payload[f] for f in MODEL_FEATURES}])
        X = _prepare_model_input(row)
        pred = model.predict(X)[0]
        return {
            "prediction": float(pred),
            "model": "sound_realty_knn",
            "model_features": MODEL_FEATURES,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /predict_required")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_sales")
def predict_sales(payload: Dict[str, Any] = Body(...)):
    """
    Minimal endpoint for sales team: accepts the sales subset + zipcode and returns prediction.
    Required fields: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, zipcode
    """
    try:
        missing = [f for f in SALE_FEATURES if f not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing sale features: {missing}")

        # Build DataFrame with only sale features (including zipcode)
        row = pd.DataFrame([{k: payload[k] for k in SALE_FEATURES}])

        # Merge demographics and prepare features
        merged = _merge_demographics(row)
        X = _prepare_model_input(merged)

        pred = model.predict(X)[0]
        return {
            "prediction": float(pred),
            "model": "sound_realty_knn",
            "model_features": MODEL_FEATURES,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /predict_sales")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
def reload_model():
    """
    Reload model and features from disk. Useful after retraining or model swap.
    (No auth here — in production add admin auth.)
    """
    global model, MODEL_FEATURES, DEMOGRAPHICS_DF
    try:
        model, MODEL_FEATURES = load_model_and_features()
        DEMOGRAPHICS_DF = load_demographics()
        logger.info("Model, features, and demographics reloaded from disk.")
        return {"status": "reloaded", "n_features": len(MODEL_FEATURES)}
    except Exception as e:
        logger.exception("Failed to reload model")
        raise HTTPException(status_code=500, detail=str(e))
