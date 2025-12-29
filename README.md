Credit Card Fraud Detection
End-to-end machine learning project to detect fraudulent credit card transactions, built to handle highly imbalanced data and provide actionable fraud risk scores for each transaction.

Overview
This project implements a complete fraud detection pipeline with A/B testing capabilities:

ML Models: XGBoost and Random Forest classifiers achieving 0.98 ROC-AUC

A/B Testing Framework: Compare model performance with real-time traffic splitting

REST API: FastAPI backend for real-time predictions

Web UI: Streamlit interfaces for batch scoring and A/B test visualization

Batch Processing: Efficient script for scoring 284K+ transactions

Dataset
Public Kaggle credit card transactions dataset with anonymized features V1–V28, Time, Amount, and label Class (1 = fraud, 0 = normal)

284,807 transactions with 492 frauds (0.17% fraud rate)

Highly imbalanced data, requiring careful evaluation metrics

Model Performance
XGBoost (Primary Model)
ROC-AUC: 0.98

Precision: 0.84

Recall: 0.84

F1-Score: 0.84

Random Forest (A/B Test Comparison)
ROC-AUC: 0.97

More conservative fraud detection (higher false positive rate)

Key Findings
XGBoost: Lower false positive rate (0.17% avg fraud probability)

Random Forest: More conservative (1.55% avg fraud probability)

Both models perform well; XGBoost selected for production due to speed and precision balance

Project Structure
text
fraud_detection_project/
├── data/                          # Data files (not in Git)
│   ├── creditcard.csv            # Original dataset (download separately)
│   └── predictions_ab.csv        # A/B test results (150MB, gitignored)
├── notebooks/
│   └── eda.ipynb                 # EDA and model training comparison
├── src/
│   ├── fraud_model.joblib        # Current production model
│   ├── fraud_model_xgboost.joblib
│   ├── fraud_model_rf.joblib
│   ├── fraud_model_logreg.joblib
│   ├── ab_testing.py             # A/B testing logic
│   └── predict_single.py         # CLI prediction script
├── api/
│   └── main.py                   # FastAPI with A/B testing endpoints
├── ui/
│   ├── app_ab_testing.py         # Streamlit A/B test runner (slow)
│   └── app_results_viewer.py     # Streamlit A/B results viewer (fast)
├── scripts/
│   └── batch_predict.py          # Batch processing for 284K rows
├── requirements.txt              # Python dependencies
├── .gitignore                    # Exclude large files
└── README.md                     # This file
Installation
Prerequisites
Python 3.8+ (tested on Python 3.13)

pip and virtualenv

200MB+ disk space for models and dependencies

Setup Steps
1. Clone the repository
bash
git clone https://github.com/Syamchand-Chilaka/fraud-detection-project.git
cd fraud-detection-project
2. Create and activate virtual environment
macOS/Linux:

bash
python3 -m venv venv
source venv/bin/activate
Windows:

bash
python -m venv venv
venv\Scripts\activate
3. Install dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Download dataset
Download the Kaggle Credit Card Fraud Detection dataset:

Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv

Place in data/creditcard.csv

Usage
Option 1: Batch Prediction Script ⭐ RECOMMENDED FOR LARGE DATASETS
Process all 284K transactions efficiently:

bash
source venv/bin/activate
python scripts/batch_predict.py
What it does:

Loads both XGBoost and Random Forest models

Randomly assigns each transaction to model A or B (50/50 split)

Processes 284,807 rows in ~40 minutes

Saves results to data/predictions_ab.csv

Shows progress bar with tqdm

Output:

text
Loading data...
Loading models...
Both models loaded.
Running predictions on 284807 rows...
Predicting: 100%|██████████| 284807/284807 [41:20<00:00, 114.82it/s]
Saving results to: data/predictions_ab.csv
Done! Results saved with 284807 predictions.
Option 2: A/B Test Results Viewer (Streamlit) ⭐ BEST FOR VISUALIZATION
View precomputed A/B test results interactively:

bash
source venv/bin/activate
streamlit run ui/app_results_viewer.py
Open browser to http://localhost:8501

Features:

Load 284K predictions instantly (reads from CSV)

Interactive pie chart showing 50/50 traffic split

Bar chart comparing fraud detection by model

Side-by-side metrics (avg fraud probability by model)

Downloadable results CSV

Sample predictions table

Option 3: REST API with A/B Testing (FastAPI)
Start the API server:

bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
Endpoints:

GET / – Health check & model info
bash
curl http://localhost:8002/
Response:

json
{
  "message": "Fraud Detection API with A/B Testing",
  "version": "2.0",
  "status": "running",
  "models": {
    "model_a": "XGBoost",
    "model_b": "Random Forest"
  }
}
POST /predict – Score a transaction (auto A/B split)
bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
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
Response:

json
{
  "fraud_probability": 0.012,
  "fraud_prediction": 0,
  "model_used": "xgboost"
}
GET /ab-stats – View A/B test statistics
bash
curl http://localhost:8002/ab-stats
Response:

json
{
  "total_predictions": 1523,
  "model_a_count": 761,
  "model_b_count": 762,
  "model_a_avg_fraud_prob": 0.0018,
  "model_b_avg_fraud_prob": 0.0156
}
Interactive API Docs:

Swagger UI: http://localhost:8002/docs

ReDoc: http://localhost:8002/redoc

Option 4: Jupyter Notebook (EDA & Model Training)
Explore data and compare models:

bash
source venv/bin/activate
jupyter notebook
Open notebooks/eda.ipynb to:

Load and explore the credit card dataset

Compare Logistic Regression, Random Forest, and XGBoost

View confusion matrices, ROC curves, and feature importance

Train and save models

A/B Testing Framework
This project includes a production-ready A/B testing system to compare model performance.

How It Works
Traffic Splitting: Each prediction is randomly assigned to Model A (XGBoost) or Model B (Random Forest) with 50/50 probability

Logging: All predictions are logged with model assignment and fraud probability

Statistics Tracking: Real-time aggregation of predictions per model

Batch Processing: batch_predict.py processes entire dataset with A/B logic

Running A/B Tests
Method 1: Batch Script (Fast)
bash
python scripts/batch_predict.py
Processes 284K rows in ~40 minutes, then view results:

bash
streamlit run ui/app_results_viewer.py
Method 2: Live API (Real-time)
bash
# Terminal 1: Start API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8002

# Terminal 2: View stats
curl http://localhost:8002/ab-stats
A/B Test Results
From 284,807 predictions:

XGBoost: 142,856 predictions (50.2%), avg fraud prob = 0.17%

Random Forest: 141,951 predictions (49.8%), avg fraud prob = 1.55%

Insight: Random Forest is ~9x more conservative, flagging more transactions as potential fraud. XGBoost balances precision/recall better for production.

Key Technical Decisions
XGBoost over Random Forest: Better performance on imbalanced data with faster inference

A/B Testing Framework: Enables data-driven model selection with production traffic

Batch Processing: Separate script for large-scale scoring vs real-time API

Class Weighting: Used scale_pos_weight to handle extreme imbalance (99.83% normal transactions)

ROC-AUC as Primary Metric: More informative than accuracy for imbalanced classification

Joblib Serialization: Fast model persistence, industry standard for scikit-learn/XGBoost

FastAPI + Streamlit: Modern Python stack for ML deployment

Git LFS Alternative: Large files (CSV) gitignored, documented download process instead

Skills Demonstrated
Machine Learning: Binary classification, imbalanced data handling, model comparison, hyperparameter tuning

Experimentation: A/B testing framework, traffic splitting, statistical logging

Python: pandas, scikit-learn, XGBoost, joblib, tqdm

API Development: FastAPI, REST endpoints, request validation, health checks

Web Development: Streamlit, interactive dashboards, data visualization with Plotly

MLOps: Model versioning, batch inference, reproducible pipelines

Version Control: Git, GitHub, proper .gitignore for large files

Problem Solving: Optimized batch processing (40 min vs 30+ hours), debugged macOS networking issues

Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    Fraud Detection System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐          ┌──────────────────┐          │
│  │  Batch Script  │          │   FastAPI API    │          │
│  │ (40 min/284K)  │          │  (Real-time)     │          │
│  └───────┬────────┘          └────────┬─────────┘          │
│          │                             │                     │
│          ▼                             ▼                     │
│  ┌──────────────────────────────────────────────┐          │
│  │         A/B Testing Controller               │          │
│  │   • Random 50/50 traffic split               │          │
│  │   • Prediction logging                       │          │
│  │   • Statistics aggregation                   │          │
│  └──────────────┬───────────────────────────────┘          │
│                 │                                            │
│      ┌──────────┴──────────┐                               │
│      ▼                      ▼                               │
│  ┌─────────┐          ┌──────────────┐                     │
│  │XGBoost  │          │Random Forest │                     │
│  │(Model A)│          │  (Model B)   │                     │
│  │7.1 MB   │          │  1.8 MB      │                     │
│  └─────────┘          └──────────────┘                     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │       Streamlit Results Viewer                     │    │
│  │  • Load predictions_ab.csv                         │    │
│  │  • Pie chart (traffic split)                       │    │
│  │  • Bar chart (fraud detection comparison)          │    │
│  │  • Metrics dashboard                               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
Requirements
See requirements.txt for full dependencies:

text
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
fastapi>=0.95.0
uvicorn>=0.21.0
streamlit>=1.20.0
matplotlib>=3.5.0
seaborn>=0.12.0
joblib>=1.2.0
plotly>=5.0.0
tqdm>=4.60.0
Troubleshooting
Python Issues
Import errors:

bash
pip install --upgrade -r requirements.txt
Streamlit won't start:

bash
# Check virtual environment
which python  # Should show venv path

# Try explicit command
python -m streamlit run ui/app_results_viewer.py
API Issues
Port already in use (8002):

bash
# Kill process on port 8002
kill -9 $(lsof -ti:8002)

# Or use different port
uvicorn api.main:app --reload --host 0.0.0.0 --port 8003
Connection refused (macOS):

bash
# Use 0.0.0.0 instead of 127.0.0.1
uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
Git Issues
Large file rejected:

bash
# Already configured in .gitignore:
# data/predictions_ab.csv
# data/*.csv
Future Enhancements
 Deploy to cloud (Render, Railway, Streamlit Cloud)

 Add model monitoring dashboard (prediction drift, performance decay)

 Implement feature importance visualizations in Streamlit

 Add confidence intervals for A/B test results

 Create Docker Compose setup for multi-container deployment

 Add CI/CD pipeline with GitHub Actions

 Implement threshold tuning UI (adjust fraud cutoff dynamically)

 Add email/SMS alerts for high-risk transactions

 Create Kubernetes deployment manifests

Author
Syamchand Chilaka
Data Scientist | ML Engineer

Built as an end-to-end ML project demonstrating:

Production ML pipelines with A/B testing

Batch and real-time inference systems

Interactive web applications and REST APIs

Handling imbalanced datasets (0.17% fraud rate)

License
MIT License

References
Dataset: Kaggle Credit Card Fraud Detection

XGBoost: XGBoost Documentation

FastAPI: FastAPI Documentation

Streamlit: Streamlit Documentation

A/B Testing: Practical Statistics for Data Scientists

⭐ Star this repo if you find it useful for learning ML deployment and A/B testing!