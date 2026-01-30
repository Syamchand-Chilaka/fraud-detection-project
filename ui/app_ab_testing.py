import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Fraud Detection A/B Test", layout="wide")

st.title("Credit Card Fraud Detection with A/B Testing")

API_URL = "http://localhost:8002"

# Sidebar for stats
st.sidebar.title("A/B Test Stats")

try:
    stats_response = requests.get(f"{API_URL}/ab-stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()

        if "message" in stats:
            st.sidebar.info("No predictions yet")
        else:
            st.sidebar.metric("Total Predictions", stats["total_predictions"])

            col1, col2 = st.sidebar.columns(2)
            col1.metric("XGBoost", stats["model_a_count"])
            col2.metric("Random Forest", stats["model_b_count"])

            st.sidebar.write("**Average Fraud Probability**")
            col1, col2 = st.sidebar.columns(2)
            col1.metric("XGBoost", f"{stats['model_a_avg_fraud_prob']:.4f}")
            col2.metric("Random Forest",
                        f"{stats['model_b_avg_fraud_prob']:.4f}")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"API not running: {str(e)}")

# Main area
st.write("## Upload Transactions")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} transactions")
    st.dataframe(df.head())

    if st.button("Run Predictions"):
        progress = st.progress(0)
        results = []

        for i, row in df.iterrows():
            try:
                response = requests.post(
                    f"{API_URL}/predict", json=row.to_dict(), timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    results.append(result)
            except requests.exceptions.RequestException as e:
                st.error(f"API connection failed: {str(e)}")
                break

            progress.progress((i + 1) / len(df))

        if results:
            results_df = pd.DataFrame(results)
            df["model_used"] = results_df["model_used"]
            df["fraud_probability"] = results_df["fraud_probability"]
            df["fraud_prediction"] = results_df["fraud_prediction"]

            st.success("Done!")

            # Show model usage
            st.write("## Model Usage")
            model_counts = df["model_used"].value_counts()
            fig = px.pie(values=model_counts.values, names=model_counts.index)
            st.plotly_chart(fig)

            # Show results
            st.write("## Results")
            st.dataframe(df[["Time", "Amount", "model_used",
                         "fraud_probability", "fraud_prediction"]].head(20))

            # Download
            csv = df.to_csv(index=False)
            st.download_button("Download Results", csv, "results.csv")
