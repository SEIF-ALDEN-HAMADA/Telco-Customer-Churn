import os
import joblib
import pandas as pd
from preprocess import preprocess_data

def predict_churn(filepath):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")
    model = joblib.load(model_path)

    X, y = preprocess_data(filepath)
    preds = model.predict(X)

    results = pd.DataFrame({
        "CustomerID": range(1, len(preds) + 1),
        "Actual": y,
        "Predicted": preds
    })

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "predictions.csv")
    results.to_csv(results_path, index=False)
    return results
