import joblib
import pandas as pd

model = joblib.load("src/fraud_model.joblib")


def predict_from_csv(path):
    df = pd.read_csv(path)

    df_features = df.copy()

    if "Class" in df_features.columns:
        df_features = df_features.drop("Class", axis=1)

        probs = model.predict_proba(df_features)[:, 1]
        preds = (probs >= 0.5).astype(int)

    df["fraud_probability"] = probs
    df["foaud_predictions"] = preds
    return df


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    out = predict_from_csv(input_path)
    out.to_csv("data/predictions.csv", index=False)
    print("Saved predictions to data/predections.csv")
