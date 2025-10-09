#!/usr/bin/env python3
"""
evaluate_model.py
-----------------
Evaluates the trained house price model (kc_house_data.csv) and 
optionally predicts on unseen data (future_unseen_examples.csv).

Outputs:
- RMSE and R² metrics on holdout/test set
- Cross-validation R²
- Optional preview of predictions on future_unseen_examples.csv
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.pkl"
FEATURES_PATH = ROOT / "model" / "model_features.json"
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
DEMOGRAPHICS_PATH = ROOT / "data" / "zipcode_demographics.csv"
FUTURE_PATH = ROOT / "data" / "future_unseen_examples.csv"

# -------------------------------------------------------------
# Load model and metadata
# -------------------------------------------------------------
print("[INFO] Loading model and feature metadata...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    MODEL_FEATURES = json.load(f)

# -------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------
print("[INFO] Loading main dataset and demographics...")
df = pd.read_csv(DATA_PATH, dtype={"zipcode": str})
demo = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})

# Merge demographics into main dataset
df = df.merge(demo, on="zipcode", how="left")

# Fill missing demographics with medians
for col in demo.columns:
    if col == "zipcode":
        continue
    if col not in df.columns:
        df[col] = demo[col].median()
    else:
        df[col] = df[col].fillna(demo[col].median())

# Drop zipcode (model doesn't use it directly)
if "zipcode" in df.columns:
    df = df.drop(columns=["zipcode"])

# -------------------------------------------------------------
# Prepare features and target
# -------------------------------------------------------------
target_col = "price"  # target column for house price prediction
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in {DATA_PATH}")

# Ensure all model features exist
for f in MODEL_FEATURES:
    if f not in df.columns:
        df[f] = 0.0

X = df[MODEL_FEATURES]
y = df[target_col]

# -------------------------------------------------------------
# Train/Test split
# -------------------------------------------------------------
print("[INFO] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------
# Evaluate performance
# -------------------------------------------------------------
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Results:")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2:.4f}")

# -------------------------------------------------------------
# Cross-validation (optional)
# -------------------------------------------------------------
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"\n Cross-Validation R²: Mean={cv_scores.mean():.4f}, Std={cv_scores.std():.4f}")

# -------------------------------------------------------------
# Optional: Predict on future unseen examples
# -------------------------------------------------------------
if FUTURE_PATH.exists():
    print("\n [INFO] Running inference on future_unseen_examples.csv...")
    future = pd.read_csv(FUTURE_PATH, dtype={"zipcode": str})
    # Merge demographics
    future = future.merge(demo, on="zipcode", how="left")
    for c in demo.columns:
        if c == "zipcode":
            continue
        if c not in future.columns:
            future[c] = demo[c].median()
        else:
            future[c] = future[c].fillna(demo[c].median())
    if "zipcode" in future.columns:
        future = future.drop(columns=["zipcode"])

    # Align with model features
    for f in MODEL_FEATURES:
        if f not in future.columns:
            future[f] = 0.0
    X_future = future[MODEL_FEATURES]

    preds = model.predict(X_future)
    out_df = pd.DataFrame({
        "prediction": preds
    })
    print("\n Sample Predictions (future_unseen_examples.csv):")
    print(out_df.head())
    out_df.to_csv(ROOT / "data" / "future_predictions.csv", index=False)
    print("[SAVED] Predictions written to data/future_predictions.csv")

else:
    print("[INFO] No future_unseen_examples.csv found — skipping inference.")
