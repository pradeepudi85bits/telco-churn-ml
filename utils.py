"""Shared helpers: cleaning + preprocessing pipeline used by training and the app."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def clean_data(df: pd.DataFrame):
    """Return (X, y). y is None when the Churn column is absent."""
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    # TotalCharges is text with blanks for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    y = None
    if TARGET in df.columns:
        y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
        df = df.drop(columns=[TARGET])
    return df[NUMERIC_COLS + CATEGORICAL_COLS], y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])
