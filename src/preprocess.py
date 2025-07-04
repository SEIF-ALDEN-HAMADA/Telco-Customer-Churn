"""
preprocess.py

Contains preprocessing logic to clean and prepare the Telco Customer Churn dataset
for machine learning.

Functions:
----------
- preprocess_data(df): Takes a raw DataFrame and returns a cleaned, encoded, and ready-to-train version.

Steps include:
--------------
- Handling missing values
- Encoding categorical features
- Converting data types
- Removing irrelevant columns (e.g., customerID)
"""

import pandas as pd
...


import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    #Drop customerID (not useful)
    df.drop("customerID", axis=1, inplace=True)

    #Replace spaces with NaN and drop missing
    df.replace(" ", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert TotalCharges to float
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    #Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y
