import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import random
import sys

DATA_PATH = Path("data/creditcard.csv")
MODEL_A_PATH = Path("src/fraud_model_xgboost.joblib")
MODEL_B_PATH = Path("src/fraud_model_rf.joblib")
OUTPUT_PATH = Path("data/predictions_ab.csv")

# Validate file existence
if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
    sys.exit(1)
if not MODEL_A_PATH.exists():
    print(f"Error: Model A not found at {MODEL_A_PATH}")
    sys.exit(1)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])

print("Loading models...")
model_a = joblib.load(MODEL_A_PATH)
try:
    model_b = joblib.load(MODEL_B_PATH)
    print("Both models loaded.")
except:
    print("Random Forest model not found, using XGBoost only.")
    model_b = model_a

print(f"Running predictions on {len(df)} rows...")

# Create A/B split with 50/50 ratio
split = random.choices(['A', 'B'], weights=[0.5, 0.5], k=len(df))

probas = []
models_used = []

for i in tqdm(range(len(X)), desc="Predicting"):
    row = X.iloc[i:i+1]
    if split[i] == 'A':
        prob = model_a.predict_proba(row)[0, 1]
        models_used.append('xgboost')
    else:
        prob = model_b.predict_proba(row)[0, 1]
        models_used.append('random_forest')
    probas.append(prob)

df["fraud_probability"] = probas
df["fraud_prediction"] = (df["fraud_probability"] >= 0.5).astype(int)
df["model_used"] = models_used

print(f"Saving results to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Done! Results saved with {len(df)} predictions.")
