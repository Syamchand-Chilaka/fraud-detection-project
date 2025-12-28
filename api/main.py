from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import FastAPI


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


@app.post("/predict")
def predict(tx: Transaction):
    # load model, preprocess, return prediction
    return {"fraud_probability": 0.12, "fraud_prediction": 0}
