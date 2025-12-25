# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card transactions, built to handle highly imbalanced data and provide actionable fraud risk scores for each transaction.

## Dataset

- Public credit card transactions dataset with anonymized features `V1`–`V28`, `Time`, `Amount`, and label `Class` (1 = fraud, 0 = normal).
- Strong class imbalance: frauds are a very small fraction of total transactions, which makes the problem realistic and challenging.

## Model

- Trained binary classifiers (e.g., Logistic Regression and Random Forest) using scikit-learn.
- Handled class imbalance using class weights / resampling.
- Evaluated using:
  - ROC-AUC
  - Precision and Recall
  - Confusion matrix

(Replace this later with real numbers, for example: “Achieved ROC-AUC of 0.98 and recall of 0.90 on the test set.”)

## Project Structure

- `data/` – input CSV files (not tracked in Git: `creditcard.csv`, `predictions.csv`).
- `notebooks/` – exploratory data analysis (`eda.ipynb`).
- `src/` – trained model artifact (`fraud_model.joblib`) and prediction script (`predict_single.py`).
- `api/` – FastAPI application exposing the model as a REST API.
- `main.py` – main entry point (if used).

## How to Run (CLI)

1. Clone the repository:

git clone https://github.com/Syamchand-Chilaka/fraud-detection-project.git
cd fraud-detection-project


2. Create and activate a virtual environment (example for macOS/Linux): python3 -m venv venv
source venv/bin/activate


3. Install dependencies:pip install -r requirements.txt


4. Download the credit card fraud dataset (`creditcard.csv`) into the `data/` folder (not included in the repo because of size).

5. Run the prediction script: python src/predict_single.py data/creditcard.csv


This will create:

- `data/predictions.csv` with all original columns plus:
  - `fraud_probability` – model score between 0 and 1.
  - `fraud_prediction` – binary prediction (1 = predicted fraud, 0 = predicted normal).

## Example Output (Batch)

The `data/predictions.csv` file contains:

- All original columns from `creditcard.csv`.
- Two additional columns:
  - `fraud_probability`
  - `fraud_prediction`

For most rows where `Class = 1` (true fraud), the model assigns high `fraud_probability` and predicts `fraud_prediction = 1`, matching the evaluation metrics seen during testing.

## REST API (FastAPI)

This project exposes the trained fraud detection model as a REST API using FastAPI.

### How to start the API

From the project root, with the virtual environment activated: uvicorn api.main:app --reload --port 8001


The API will be available at:

- Base URL: <http://127.0.0.1:8001>
- Interactive docs (Swagger UI): <http://127.0.0.1:8001/docs>

### Endpoints

#### `GET /`

Health check endpoint.

- **Description:** Returns a simple JSON message confirming the API is running.
- **Response example:**

{
"message": "Fraud Detection API is running"
}


#### `POST /predict`

Score a single credit card transaction.

- **Request body:** JSON matching the transaction schema (`Time`, `V1`–`V28`, `Amount`), for example:

{
"Time": 0.0,
"V1": 0.0,
"V2": 0.0,
"V3": 0.0,
"V4": 0.0,
"V5": 0.0,
"V6": 0.0,
"V7": 0.0,
"V8": 0.0,
"V9": 0.0,
"V10": 0.0,
"V11": 0.0,
"V12": 0.0,
"V13": 0.0,
"V14": 0.0,
"V15": 0.0,
"V16": 0.0,
"V17": 0.0,
"V18": 0.0,
"V19": 0.0,
"V20": 0.0,
"V21": 0.0,
"V22": 0.0,
"V23": 0.0,
"V24": 0.0,
"V25": 0.0,
"V26": 0.0,
"V27": 0.0,
"V28": 0.0,
"Amount": 0.0
}


- **Response example:**
{
"fraud_probability": 0.12,
"fraud_prediction": 0
}


Where:

- `fraud_probability` is the model’s estimated probability that the transaction is fraudulent (between 0 and 1).
- `fraud_prediction` is the binary decision (1 = predicted fraud, 0 = predicted normal).

## Future Work

- Deploy the FastAPI service behind a production server (e.g., Docker + cloud).
- Add a small web UI (Streamlit or React) to upload CSVs and visualize high-risk transactions.
- Experiment with more advanced models (e.g., XGBoost, LightGBM) and anomaly detection approaches.







