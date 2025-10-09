#!/usr/bin/env python3
"""
Sound Realty House Price Prediction API

Implements a FastAPI app that:
- Accepts house data JSON payloads (without demographics)
- Adds zipcode-level demographics automatically
- Returns model prediction and metadata

Endpoints:
    GET  /               : Welcome message
    GET  /health         : Health + model metadata
    POST /predict        : Main endpoint for unseen house data, Predict house price for unseen
    POST /predict_required : Predict when caller provides all required model features directly
    POST /reload_model   : Reloads model and demographics from disk

Artifacts expected:
    model/model.pkl
    model/model_features.json
    data/zipcode_demographics.csv
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

# -----------------------------
# Configuration
# -----------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.pkl"
FEATURES_PATH = ROOT / "model" / "model_features.json"
DEMOGRAPHICS_PATH = ROOT / "data" / "zipcode_demographics.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sound_realty_api")

# -----------------------------
# Load model and metadata
# -----------------------------
def load_model_and_features():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    return model, features


def load_demographics():
    if not DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(f"Demographics CSV not found at: {DEMOGRAPHICS_PATH}")
    return pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})


# Initialize at startup
try:
    model, MODEL_FEATURES = load_model_and_features()
    DEMOGRAPHICS_DF = load_demographics()
    logger.info(f"Model and demographics loaded. Features: {len(MODEL_FEATURES)}")
except Exception as e:
    logger.exception("Startup load failed.")
    raise

# -----------------------------
# App definition
# -----------------------------
app = FastAPI(
    title="Sound Realty House Price Predictor",
    version="1.1",
    description="Predicts house prices using trained model + zipcode demographics."
)

@app.get("/")
def root():
    return {"message": "Sound Realty Price Prediction API is running."}


@app.get("/health")
def health():
    """Return system and model metadata."""
    return {
        "status": "ok",
        "model_file": str(MODEL_PATH),
        "n_features": len(MODEL_FEATURES),
        "features_sample": MODEL_FEATURES[:5]
    }

# -----------------------------
# Helper functions
# -----------------------------
def merge_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Merge zipcode-level demographics onto the input DataFrame."""
    if "zipcode" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'zipcode' field in input.")
    
    df["zipcode"] = df["zipcode"].astype(str)
    merged = df.merge(DEMOGRAPHICS_DF, on="zipcode", how="left")

    # Fill missing demographic columns with medians
    for c in DEMOGRAPHICS_DF.columns:
        if c == "zipcode":
            continue
        if c not in merged.columns:
            merged[c] = DEMOGRAPHICS_DF[c].median()
        else:
            merged[c] = merged[c].fillna(DEMOGRAPHICS_DF[c].median())

    # Drop zipcode after merge (model doesn’t use it)
    if "zipcode" in merged.columns:
        merged = merged.drop(columns=["zipcode"])
    return merged


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure input columns match model expectations."""
    for f in MODEL_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    X = df[MODEL_FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    """
    Predict house price for unseen example.
    Input: Columns from future_unseen_examples.csv + zipcode
    Output: Predicted price + model metadata
    """
    try:
        row = pd.DataFrame([payload])
        merged = merge_demographics(row)
        X = prepare_input(merged)
        pred = model.predict(X)[0]

        return {
            "prediction": float(pred),
            "model": "GradientBoostingRegressor",
            "n_features": len(MODEL_FEATURES)
        }
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_required")
def predict_required(payload: Dict[str, Any] = Body(...)):
    """Predict when caller provides all required model features directly."""
    missing = [f for f in MODEL_FEATURES if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    try:
        row = pd.DataFrame([{f: payload[f] for f in MODEL_FEATURES}])
        X = prepare_input(row)
        pred = model.predict(X)[0]
        return {"prediction": float(pred), "model": "GradientBoostingRegressor"}
    except Exception as e:
        logger.exception("Prediction (required) failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
def reload_model():
    """Reload model and demographics from disk (for hot updates)."""
    global model, MODEL_FEATURES, DEMOGRAPHICS_DF
    try:
        model, MODEL_FEATURES = load_model_and_features()
        DEMOGRAPHICS_DF = load_demographics()
        logger.info("Model and demographics reloaded.")
        return {"status": "reloaded", "n_features": len(MODEL_FEATURES)}
    except Exception as e:
        logger.exception("Reload failed.")
        raise HTTPException(status_code=500, detail=str(e))
