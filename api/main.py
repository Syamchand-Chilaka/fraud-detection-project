from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = "src/fraud_model.joblib"
model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
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
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    # im converting incoming JSON to DataFrame with one row
    data = pd.DataFrame([transaction.dict()])

    # Model outputs probability for class 1 (fraud)
    proba = model.predict_proba(data)[0][1]
    pred = int(proba >= 0.5)

    return {
        "fraud_probability": float(proba),
        "fraud_prediction": pred

    }
