#!/usr/bin/env python3
"""
test_client.py

Simple test script that posts example rows from data/future_unseen_examples.csv
to the running FastAPI service endpoints.

Usage:
  1) Start the API:
       uvicorn app:app --reload
  2) Run this script:
       python test_client.py
"""

import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FUTURE_CSV = DATA_DIR / "future_unseen_examples.csv"

API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
PREDICT_SALES = f"{API_BASE}/predict_sales"
PREDICT_REQUIRED = f"{API_BASE}/predict_required"

# How many examples to try
N_EXAMPLES = 3


def post_examples():
    df = pd.read_csv(FUTURE_CSV, dtype={"zipcode": str})
    examples = df.head(N_EXAMPLES).to_dict(orient="records")

    print(f"Posting {len(examples)} example(s) to {PREDICT_URL}")
    for ex in examples:
        r = requests.post(PREDICT_URL, json=ex, timeout=10)
        print("status:", r.status_code, r.json())

    # Also post a minimal sales example to /predict_sales
    sample_sale = {
        "bedrooms": 3,
        "bathrooms": 1.75,
        "sqft_living": 1710,
        "sqft_lot": 5000,
        "floors": 1,
        "sqft_above": 1710,
        "sqft_basement": 0,
        "zipcode": "98178",
    }
    print(f"\nPosting sample to {PREDICT_SALES}")
    r = requests.post(PREDICT_SALES, json=sample_sale, timeout=10)
    print("status:", r.status_code, r.json())

    # To call /predict_required, build a payload with the model_features order.
    # The model_features.json file can be loaded to construct such a payload.
    try:
        features = json.load(open(ROOT / "model" / "model_features.json"))
        print("\nPosting to /predict_required with zeros for every feature (demo):")
        req_payload = {f: 0.0 for f in features}
        r = requests.post(PREDICT_REQUIRED, json=req_payload, timeout=10)
        print("status:", r.status_code, r.json())
    except Exception as e:
        print("Could not load model_features.json to demo /predict_required:", e)


if __name__ == "__main__":
    post_examples()
