import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import shutil

print("=" * 60)
print("TRAINING MODELS FOR A/B TESTING")
print("=" * 60)

print("\n[1/5] Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f" Loaded {len(df):,} transactions")

print("\n[2/5] Preparing train/test split...")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f" Training set: {len(X_train):,} samples")
print(f" Test set: {len(X_test):,} samples")

print("\n[3/5] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_depth=10
)
rf.fit(X_train, y_train)
joblib.dump(rf, "src/fraud_model_rf.joblib")
print(" Saved: src/fraud_model_rf.joblib")

# Training Logistic Regression
print("\n[4/5] Training Logistic Regression...")
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
lr.fit(X_train, y_train)
joblib.dump(lr, "src/fraud_model_logreg.joblib")
print(" Saved: src/fraud_model_logreg.joblib")

print("\n[5/5] Copying XGBoost model...")
shutil.copy("src/fraud_model.joblib", "src/fraud_model_xgboost.joblib")
print("✓ Saved: src/fraud_model_xgboost.joblib")

print("\n" + "=" * 60)
print(" ALL MODELS SAVED FOR A/B TESTING")
print("=" * 60)
print("\nSaved models:")
print("  • fraud_model_xgboost.joblib  (Model A)")
print("  • fraud_model_rf.joblib       (Model B)")
print("  • fraud_model_logreg.joblib   (Baseline)")
print("\nReady for Phase 2: Create ab_testing.py")
