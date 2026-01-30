from src.ab_testing import ABTestingManager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
from pathlib import Path
sys.path.append("src")


app = FastAPI(title="Fraud Detection API with A/B Testing",
              description="Compare XGBoost vs Random Forest models in production",
              version="2.0"
              )

# Run with: uvicorn api.main:app --reload --port 8002

# Validate models exist before initializing
model_a_path = "src/fraud_model_xgboost.joblib"
model_b_path = "src/fraud_model_rf.joblib"

if not Path(model_a_path).exists():
    raise FileNotFoundError(f"Model A not found at {model_a_path}. Please train models first.")
if not Path(model_b_path).exists():
    raise FileNotFoundError(f"Model B not found at {model_b_path}. Please train models first.")

ab_manager = ABTestingManager(
    model_a_path=model_a_path,
    model_b_path=model_b_path,
    split_ratio=0.5
)


class Transaction(BaseModel):
    """Credit Card Transaction Schema"""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def root():
    """"Health check endpoint"""
    return {
        "message": "Fraud Detection API with A/B Testing",
        "version": "2.0",
        "status": "running",
        "models": {
            "model_a": "XGBoost",
            "model_b": "Random Forest"
        }
    }


@app.post("/predict")
def predict(trans: Transaction):
    """Predict fraud using A/B testing"""
    # Convert to DataFrame
    df = pd.DataFrame([trans.model_dump()])
    result = ab_manager.predict(df)

    return result


@app.get("/ab-stats")
def ab_stats():
    """Get A/B testing statistics"""
    return ab_manager.get_stats()


@app.post("/ab-clear-logs")
def ab_clear_logs():
    """Clear all A/B test logs"""
    result = ab_manager.clear_logs()
    return result
