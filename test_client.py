#!/usr/bin/env python3
"""
test_api.py
-----------
Simple client to test the Sound Realty API.

Demonstrates that the /predict and /predict_required endpoints work correctly.
"""

import requests
import pandas as pd
import json

API_URL = "http://127.0.0.1:8000"  # Update with deployed URL if hosted
PREDICT_ENDPOINT = f"{API_URL}/predict"
PREDICT_REQUIRED_ENDPOINT = f"{API_URL}/predict_required"

# Load example data
data = pd.read_csv("data/future_unseen_examples.csv")
example = data.iloc[0].to_dict()

print("[INFO] Sending example to /predict ...")
resp = requests.post(PREDICT_ENDPOINT, json=example)

if resp.status_code == 200:
    print("/predict response:")
    print(json.dumps(resp.json(), indent=2))
else:
    print("/predict failed:", resp.status_code, resp.text)

# Optional: test /predict_required if you have feature-only JSON
print("\n[INFO] Sending to /predict_required ...")
try:
    resp2 = requests.post(PREDICT_REQUIRED_ENDPOINT, json=example)
    print("/predict_required response:")
    print(json.dumps(resp2.json(), indent=2))
except Exception as e:
    print("Error:", e)
