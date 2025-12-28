# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card transactions, built to handle highly imbalanced data and provide actionable fraud risk scores for each transaction.

## Overview

This project implements a complete fraud detection pipeline:
- **ML Model**: XGBoost binary classifier achieving 0.98 ROC-AUC
- **REST API**: FastAPI backend for real-time predictions
- **Web UI**: Streamlit interface for batch transaction scoring

## Dataset

- Public Kaggle credit card transactions dataset with anonymized features `V1`–`V28`, `Time`, `Amount`, and label `Class` (1 = fraud, 0 = normal).
- **284,807 transactions** with **492 frauds** (0.17% fraud rate)
- Highly imbalanced data, requiring careful evaluation metrics

## Model

- **Final Model**: XGBoost Classifier
- **Performance** (on test set):
  - ROC-AUC: **0.98**
  - Precision: **0.84**
  - Recall: **0.84**
  - F1-Score: **0.84**

- **Baseline Comparison**: Tested Logistic Regression and Random Forest; XGBoost performed best on imbalanced data
- **Handling Imbalance**: Used `class_weight='balanced'` to handle 99.8% legitimate transactions

## Project Structure

fraud_detection_project/
├── data/ # Input CSV files (not in Git)
│ └── creditcard.csv # Full dataset (150.8 MB)
├── notebooks/
│ └── eda.ipynb # Exploratory data analysis & model training
├── src/
│ ├── fraud_model.joblib # Trained XGBoost model
│ └── predict_single.py # CLI prediction script
├── api/
│ └── main.py # FastAPI backend
├── ui/
│ └── app_standalone.py # Streamlit web interface
├── requirements.txt # Python dependencies
└── README.md # This file

text

## Installation

### 1. Clone the repository

git clone https://github.com/Syamchand-Chilaka/fraud-detection-project.git
cd fraud-detection-project

text

### 2. Create and activate virtual environment

**macOS/Linux:**
python3 -m venv venv
source venv/bin/activate

text

**Windows:**
python -m venv venv
venv\Scripts\activate

text

### 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

text

### 4. Download dataset

Download the Kaggle Credit Card Fraud Detection dataset and place `creditcard.csv` in the `data/` folder:
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Then move to:
data/creditcard.csv

text

---

## Usage

### Option 1: Web UI (Streamlit) ⭐ **RECOMMENDED**

Start the interactive Streamlit app to upload CSV and see predictions:

source venv/bin/activate
streamlit run ui/app_standalone.py

text

Then open your browser to `http://localhost:8501` and:
1. Click **Browse files**
2. Upload a CSV with transactions (same schema as training data)
3. Click **Score transactions**
4. View predictions table with `fraud_probability` and `fraud_prediction` columns

**Example output:**
Processed 284,807 transactions

Time	Amount	fraud_probability	fraud_prediction
0	149.62	0.002	0
1	2.69	0.001	0
2	378.66	0.450	0
text

---

### Option 2: REST API (FastAPI)

Start the API server:

source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8002

text

The API will be available at:
- **Base URL**: `http://localhost:8002`
- **Interactive Docs**: `http://localhost:8002/docs` (Swagger UI)

#### Endpoints

**GET `/`** – Health check
curl http://localhost:8002/

text

Response:
{
"message": "Fraud Detection API is running"
}

text

**POST `/predict`** – Score a single transaction

Request:
curl -X POST "http://localhost:8002/predict"
-H "Content-Type: application/json"
-d '{
"Time": 0,
"V1": -1.3598,
"V2": -0.0728,
"V3": 2.5363,
"V4": 1.3782,
"V5": -0.3383,
"V6": 0.4624,
"V7": 0.2396,
"V8": 0.0987,
"V9": 0.3638,
"V10": 0.0908,
"V11": -0.5516,
"V12": -0.6178,
"V13": -0.9914,
"V14": -0.3112,
"V15": 1.4682,
"V16": -0.4704,
"V17": 0.208,
"V18": 0.0258,
"V19": 0.404,
"V20": 0.2514,
"V21": -0.0183,
"V22": 0.2778,
"V23": -0.1105,
"V24": 0.0669,
"V25": 0.1286,
"V26": -0.1891,
"V27": 0.1335,
"V28": -0.0211,
"Amount": 149.62
}'

text

Response:
{
"fraud_probability": 0.002,
"fraud_prediction": 0
}

text

Where:
- `fraud_probability` (0–1): Model's estimated probability that transaction is fraudulent
- `fraud_prediction` (0 or 1): Binary classification (1 = fraud, 0 = normal)

---

### Option 3: Jupyter Notebook (EDA & Training)

Explore the data and retrain the model:

source venv/bin/activate
jupyter notebook

text

Open `notebooks/eda.ipynb` to:
- Load and explore the credit card dataset
- Compare Logistic Regression, Random Forest, and XGBoost
- View confusion matrices and ROC curves
- Save the trained model

---

### Option 4: CLI Batch Scoring

Score a CSV of transactions using the command line:

source venv/bin/activate
python src/predict_single.py data/creditcard.csv

text

This creates `data/predictions.csv` with all original columns plus:
- `fraud_probability` – Model score (0–1)
- `fraud_prediction` – Binary prediction (0 or 1)

---

## Key Technical Decisions

1. **XGBoost over Random Forest**: Better performance on imbalanced data with faster inference
2. **Class Weighting**: Used `class_weight='balanced'` to handle extreme imbalance without data loss
3. **ROC-AUC as Primary Metric**: More informative than accuracy for imbalanced classification
4. **Joblib Serialization**: Standard in ML for fast model persistence
5. **Standalone Streamlit**: Direct model loading avoids network complexity and improves reliability

## Skills Demonstrated

- **Machine Learning**: Model training, evaluation, hyperparameter tuning, class imbalance handling
- **Python**: pandas, scikit-learn, XGBoost, joblib
- **API Development**: FastAPI, REST endpoints, request validation
- **Web Development**: Streamlit, interactive UIs, file upload handling
- **MLOps**: Model versioning, reproducible pipelines, environment management
- **Version Control**: Git, GitHub, clean commit history
- **Problem Solving**: Debugged macOS networking issues, optimized model performance

## Future Enhancements

- Deploy Streamlit app to Streamlit Cloud (free)
- Deploy FastAPI to cloud (Render, Railway, AWS)
- Add model monitoring and prediction logging
- Create Docker container for easy deployment
- Add threshold tuning UI (adjust fraud cutoff dynamically)
- Implement feature importance visualizations

## Requirements

See `requirements.txt` for full list:
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
fastapi>=0.95.0
uvicorn>=0.21.0
streamlit>=1.20.0
matplotlib>=3.5.0
seaborn>=0.12.0
joblib>=1.2.0

text

## How to Reproduce

1. Clone repo
git clone https://github.com/Syamchand-Chilaka/fraud-detection-project.git
cd fraud-detection-project

2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Download dataset
Download creditcard.csv from Kaggle and place in data/
4. Run Streamlit app
streamlit run ui/app_standalone.py

5. Upload creditcard.csv and click "Score transactions"
text

## Author

Built as an end-to-end ML project for portfolio and learning.

## License

MIT License

## References

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- FastAPI: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- Streamlit: [Streamlit Documentation](https://docs.streamlit.io/)