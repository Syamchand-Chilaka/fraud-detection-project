import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Detection A/B Results", layout="wide")
st.title("Credit Card Fraud Detection - A/B Test Results")

# Load precomputed results
@st.cache_data
def load_results():
    return pd.read_csv("data/predictions_ab.csv")

df = pd.read_csv("data/predictions_ab.csv")

st.success(f"Loaded {len(df):,} predictions")

# Sidebar stats
st.sidebar.title("A/B Test Summary")

model_counts = df["model_used"].value_counts()
st.sidebar.metric("XGBoost predictions", model_counts.get("xgboost", 0))
st.sidebar.metric("Random Forest predictions", model_counts.get("random_forest", 0))

# Average fraud probability by model
xgb_avg = df[df["model_used"] == "xgboost"]["fraud_probability"].mean()
rf_avg = df[df["model_used"] == "random_forest"]["fraud_probability"].mean()

st.sidebar.write("**Avg Fraud Probability**")
st.sidebar.metric("XGBoost", f"{xgb_avg:.4f}")
st.sidebar.metric("Random Forest", f"{rf_avg:.4f}")

# Model usage pie chart
st.write("## Model Usage Distribution")
fig = px.pie(values=model_counts.values, names=model_counts.index, 
             title="A/B Test Traffic Split")
st.plotly_chart(fig)

# Fraud detection comparison
st.write("## Fraud Detection by Model")
fraud_by_model = df.groupby("model_used")["fraud_prediction"].sum()
fig2 = px.bar(x=fraud_by_model.index, y=fraud_by_model.values,
              labels={"x": "Model", "y": "Fraud Cases Detected"},
              title="Total Fraud Cases Detected")
st.plotly_chart(fig2)

# Sample predictions
st.write("## Sample Predictions")
st.dataframe(df[["Time", "Amount", "Class", "fraud_probability", 
                 "fraud_prediction", "model_used"]].head(100))

# Download button
csv = df.to_csv(index=False)
st.download_button("Download Full Results", csv, "predictions_ab.csv")

