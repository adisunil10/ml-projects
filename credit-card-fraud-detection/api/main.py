"""
Run with: uvicorn api.main:app --reload
  (from the credit-card-fraud-detection/ directory)

POST /predict  — score a single transaction
GET  /health   — liveness check
"""

import json
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost.joblib"
HOUR_STATS_PATH = BASE_DIR / "models" / "hour_stats.json"

app = FastAPI(title="Fraud Detection API")

model = joblib.load(MODEL_PATH)
with open(HOUR_STATS_PATH) as f:
    hour_stats = json.load(f)

# Threshold from cost-minimisation analysis (run pipeline.py to update)
DECISION_THRESHOLD = 0.963


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float


def build_features(txn: Transaction) -> np.ndarray:
    t = txn
    hour = int((t.Time // 3600) % 24)
    day = int((t.Time // 86400) % 7)
    amount_log = np.log1p(max(t.Amount, 0))

    h_mean = hour_stats["amount_mean"].get(str(hour), t.Amount)
    h_std = hour_stats["amount_std"].get(str(hour), 1.0) or 1.0
    amount_z = (t.Amount - h_mean) / h_std

    # Velocity features can't be computed without historical context at inference time.
    # Using 0 as a neutral placeholder — in production this would come from a feature store.
    velocity_count = 0.0
    velocity_amount = 0.0

    return np.array([[
        t.Time, t.Amount,
        t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7, t.V8, t.V9, t.V10,
        t.V11, t.V12, t.V13, t.V14, t.V15, t.V16, t.V17, t.V18, t.V19, t.V20,
        t.V21, t.V22, t.V23, t.V24, t.V25, t.V26, t.V27, t.V28,
        amount_log, hour, day,
        t.V1 * t.V2, t.V3 * t.V4, t.V1 * t.V3,
        amount_z,
        velocity_count, velocity_amount,
    ]])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    try:
        features = build_features(txn)
        proba = float(model.predict_proba(features)[0][1])
        is_fraud = proba >= DECISION_THRESHOLD
        return {
            "fraud_probability": round(proba, 4),
            "prediction": "fraud" if is_fraud else "legitimate",
            "threshold": DECISION_THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
