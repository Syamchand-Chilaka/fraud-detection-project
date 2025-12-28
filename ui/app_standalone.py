import pandas as pd
import streamlit as st
import joblib

# Load model directly (no API call)
model = joblib.load("src/fraud_model.joblib")

st.title("Credit Card Fraud Detector")

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    X = df.drop(columns=["Class"], errors="ignore")
    
    # Predict directly in Streamlit
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_prediction"] = model.predict(X)
    
    st.write(f"Processed {len(df)} transactions")
    st.dataframe(df[["Time", "Amount", "fraud_probability", "fraud_prediction"]].head(20))
