"""Data preparation for the Telco churn project (shared by training script and Streamlit app)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

LABEL = "Churn"
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

# 'gender' is deliberately left out - it carries almost no signal for churn
num_features = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "service_count"]
cat_features = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def prepare_telco_frame(raw: pd.DataFrame):
    """Clean the raw Telco CSV. Returns (features, target); target is None if Churn is missing."""
    df = raw.copy()
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges arrives as text; blanks belong to customers with tenure 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # engineered features
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["service_count"] = sum((df[c] == "Yes").astype(int) for c in ADDON_COLS)

    target = None
    if LABEL in df.columns:
        target = (df[LABEL].astype(str).str.strip().str.lower() == "yes").astype(int)
        df = df.drop(columns=[LABEL])
    return df[num_features + cat_features], target


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("scale", StandardScaler(), num_features),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])
