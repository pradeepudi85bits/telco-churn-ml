import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)
from utils import clean_data

st.set_page_config(page_title="Telco Churn Classifier Lab", page_icon="📡", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
if uploaded is None:
    st.sidebar.info("No file uploaded - using the bundled `test_data.csv`.")
    df_raw = pd.read_csv(os.path.join(BASE, "test_data.csv"))
else:
    df_raw = pd.read_csv(uploaded)

model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
threshold = st.sidebar.slider("Decision threshold (churn)", 0.10, 0.90, 0.50, 0.05)
compare_all = st.sidebar.checkbox("Show comparison of all 5 models", value=True)

# ---------------- Main ----------------
st.title("📡 Telco Customer Churn - Model Explorer")
st.caption("Predict whether a telecom customer will churn using five classic ML classifiers.")

with st.expander("Preview of uploaded data", expanded=False):
    st.write(f"{df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(20), width="stretch")

try:
    X, y = clean_data(df_raw)
except Exception as e:
    st.error(f"Could not process the file. Make sure it has the Telco churn columns. Details: {e}")
    st.stop()

if y is None:
    st.warning("No 'Churn' column found - showing predictions only (no metrics).")

pipe = load_model(model_name)
proba = pipe.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)

st.subheader(f"Results: {model_name}")

if y is not None:
    m = compute_metrics(y, pred, proba)
    cols = st.columns(6)
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, proba)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=["No churn", "Churn"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    if compare_all:
        st.subheader("All models side by side")
        rows = []
        for name in MODEL_FILES:
            p = load_model(name).predict_proba(X)[:, 1]
            rows.append({"Model": name, **compute_metrics(y, (p >= threshold).astype(int), p)})
        comp = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(comp.style.highlight_max(axis=0, color="#cfe8cf"), width="stretch")
        st.bar_chart(comp[["Accuracy", "AUC", "F1", "MCC"]])

# Predictions download
out = df_raw.copy()
out["Churn_Probability"] = proba.round(4)
out["Predicted_Churn"] = np.where(pred == 1, "Yes", "No")
st.subheader("Predictions")
st.dataframe(out.head(50), width="stretch")
st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                   "churn_predictions.csv", "text/csv")
