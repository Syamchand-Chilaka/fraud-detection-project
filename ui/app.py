import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8002") + "/predict"
st.title("Credit Card Fraud Detector")

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview of uploaded data")
    st.write(df.head())

    if st.button("Score transactions"):
        results = []
        
        try:
            for _, row in df.iterrows():
                payload = row.to_dict()
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                results.append(response.json())

            preds_df = pd.DataFrame(results)
            st.subheader("Predictions")
            st.write(preds_df)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}. Make sure the API is running at {API_URL}")
