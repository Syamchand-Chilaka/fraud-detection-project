import random
import joblib
import pandas as pd
from datetime import datetime
import json
import os


class ABTestingManager:
    """
    A/B testing manager for comparing two fraud detection models in production.
    Randomly assigns predictions to Model A or Model B and logs results.
    """

    def __init__(self, model_a_path, model_b_path, split_ratio=0.5):
        """
        Initialize A/B testing with two models.

        Parameters:
        -----------
        model_a_path : str
            Path to Model A (e.g., XGBoost)
        model_b_path : str
            Path to Model B (e.g., Random Forest)
        split_ratio : float, default=0.5
            Percentage of traffic to Model A (0.5 = 50/50 split)
        """
        self.model_a = joblib.load(model_a_path)
        self.model_b = joblib.load(model_b_path)
        self.split_ratio = split_ratio
        self.log_file = "data/ab_test_logs.json"

        # Create log file if doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def predict(self, transaction_data):
        """
        Randomly assign to Model A or B, get prediction, log result.

        Parameters:
        -----------
        transaction_data : pandas DataFrame or numpy array
            Single transaction features

        Returns:
        --------
        dict : Prediction result with model name, probability, and prediction
        """
        # Randomly assign model based on split ratio
        use_model_a = random.random() < self.split_ratio
        model_name = "xgboost" if use_model_a else "random_forest"
        model = self.model_a if use_model_a else self.model_b

        # Get prediction
        fraud_prob = float(model.predict_proba(transaction_data)[:, 1][0])
        fraud_pred = int(model.predict(transaction_data)[0])

        # Log the result
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "fraud_probability": fraud_prob,
            "fraud_prediction": fraud_pred
        }

        self._log_prediction(log_entry)

        return {
            "model_used": model_name,
            "fraud_probability": fraud_prob,
            "fraud_prediction": fraud_pred
        }

    def _log_prediction(self, entry):
        """Append prediction to log file"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        logs.append(entry)

        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_stats(self):
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"message": "No predictions logged yet"}

        if not logs:
            return {"message": "No predictions logged yet"}

        df = pd.DataFrame(logs)

        # Calculate statistics with NaN handling
        model_a_data = df[df['model_used'] == 'xgboost']
        model_b_data = df[df['model_used'] == 'random_forest']

        stats = {
            "total_predictions": len(df),
            "model_a_count": int(len(model_a_data)),
            "model_b_count": int(len(model_b_data)),
            "model_a_avg_fraud_prob": float(model_a_data['fraud_probability'].mean()) if len(model_a_data) > 0 else 0.0,
            "model_b_avg_fraud_prob": float(model_b_data['fraud_probability'].mean()) if len(model_b_data) > 0 else 0.0,
            "model_a_fraud_rate": float((model_a_data['fraud_prediction'] == 1).mean()) if len(model_a_data) > 0 else 0.0,
            "model_b_fraud_rate": float((model_b_data['fraud_prediction'] == 1).mean()) if len(model_b_data) > 0 else 0.0
        }

        return stats
