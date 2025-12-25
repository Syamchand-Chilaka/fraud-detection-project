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
  - Precision, Recall
  - Confusion matrix

(You can later fill in actual numbers, for example: “Achieved ROC-AUC of 0.xx and recall of 0.yy on the test set.”)

## Project Structure

- `data/` – input CSV files (not tracked in Git: `creditcard.csv`, `predictions.csv`).
- `notebooks/` – exploratory data analysis (`eda.ipynb`).
- `src/` – trained model artifact (`fraud_model.joblib`) and prediction script (`predict_single.py`).
- `main.py` – main entry point (if used).

## How to Run

1. Clone the repository:

git clone https://github.com/Syamchand-Chilaka/fraud-detection-project.git
cd fraud-detection-project

python3 -m venv venv
source venv/bin/activate


3. Install dependencies: pip install -r requirements.txt


4. Download the credit card fraud dataset (`creditcard.csv`) into the `data/` folder (not included in the repo because of size).

5. Run the prediction script: python src/predict_single.py data/creditcard.csv


This will create:

- `data/predictions.csv` with all original columns plus:
  - `fraud_probability` – model score between 0 and 1.
  - `fraud_prediction` – binary prediction (1 = predicted fraud, 0 = predicted normal).

## Example Output

The `data/predictions.csv` file contains:

- All original columns from `creditcard.csv`.
- Two additional columns:
- `fraud_probability`
- `fraud_prediction`

For most rows where `Class = 1` (true fraud), the model assigns high `fraud_probability` and predicts `fraud_prediction = 1`, matching the evaluation metrics seen during testing.

## Future Work

- Deploy a FastAPI/Flask REST API for online scoring.
- Add a small web UI to upload CSVs and inspect high-risk transactions.


