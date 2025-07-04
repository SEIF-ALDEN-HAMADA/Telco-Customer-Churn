"""
train.py

This script loads and preprocesses the Telco Customer Churn dataset,
then trains a Random Forest model and evaluates its performance using accuracy,
confusion matrix, and classification report.

Functions:
----------
- train_model(filepath): Reads CSV data, applies preprocessing, trains the model, and prints evaluation metrics.

Example usage:
--------------
from train import train_model
train_model("data/Telco-Customer-Churn.csv")
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
...



import os
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def train_model(filepath):
    """
    Loads the dataset, preprocesses it, trains a Random Forest classifier,
    and prints evaluation metrics.
    
    Parameters:
    -----------
    filepath: str
        Path to the Telco-Customer-Churn dataset (CSV file)
    """
    X, y = preprocess_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    #Save the model
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(model, model_path)

    #🧪Evaluation
    y_pred = model.predict(X_test)

    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 🔍  Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()