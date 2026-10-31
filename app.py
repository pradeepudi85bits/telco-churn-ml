import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)
from preprocessing import prepare_telco_frame

st.set_page_config(page_title="Churn Radar - Telco", page_icon="📉", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}


@st.cache_resource
def get_model(name):
    return joblib.load(os.path.join(HERE, "model", MODEL_PKL[name]))


def score_all(y_true, label, proba, cutoff):
    pred = (proba >= cutoff).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


@st.cache_data(show_spinner="Computing feature importance...")
def importance_table(name, data_json):
    df = pd.read_json(io.StringIO(data_json))
    Xf, yf = prepare_telco_frame(df)
    r = permutation_importance(get_model(name), Xf, yf, scoring="roc_auc",
                               n_repeats=3, random_state=7, n_jobs=1)
    return (pd.DataFrame({"feature": Xf.columns, "auc_drop": r.importances_mean})
            .sort_values("auc_drop", ascending=False))


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Setup")
upload = st.sidebar.file_uploader("Test data (CSV with a Churn column)", type="csv")
if upload is None:
    st.sidebar.caption("Nothing uploaded - falling back to the bundled test_data.csv")
    raw_df = pd.read_csv(os.path.join(HERE, "test_data.csv"))
else:
    raw_df = pd.read_csv(upload)

chosen = st.sidebar.selectbox("Model", list(MODEL_PKL))
cutoff = st.sidebar.slider("Churn probability cut-off", 0.10, 0.90, 0.50, 0.05)

st.title("📉 Churn Radar")
st.write("Score telecom customers with five tuned classifiers and see which customers are most likely to leave.")

try:
    X, y = prepare_telco_frame(raw_df)
except Exception as err:
    st.error(f"This file doesn't look like the Telco churn data: {err}")
    st.stop()

model = get_model(chosen)
proba = model.predict_proba(X)[:, 1]
pred = (proba >= cutoff).astype(int)

tab_eval, tab_cmp, tab_one, tab_imp, tab_thr = st.tabs(
    ["Evaluation", "Model comparison", "Single customer", "Feature importance", "Cut-off explorer"])

# ------------------------------------------------------------- evaluation
with tab_eval:
    st.caption(f"{len(raw_df):,} customers loaded | model: {chosen} | cut-off: {cutoff:.2f}")
    if y is None:
        st.warning("No Churn column in this file, so only predictions are shown.")
    else:
        m = score_all(y, None, proba, cutoff)
        for col, (k, v) in zip(st.columns(6), m.items()):
            col.metric(k, f"{v:.3f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Confusion matrix")
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            sns.heatmap(confusion_matrix(y, pred, labels=[0, 1]), annot=True, fmt="d",
                        cmap="Oranges", cbar=False, ax=ax,
                        xticklabels=["Stays", "Churns"], yticklabels=["Stays", "Churns"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
        with c2:
            st.markdown("##### ROC curve")
            fpr, tpr, _ = roc_curve(y, proba)
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.plot(fpr, tpr, color="#d95f02", label=f"AUC {m['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", alpha=.5)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        st.markdown("##### Classification report")
        rep = classification_report(y, pred, target_names=["Stays", "Churns"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    st.markdown("##### Highest-risk customers")
    scored = raw_df.copy()
    scored["churn_probability"] = proba.round(4)
    scored["predicted_churn"] = np.where(pred == 1, "Yes", "No")
    st.dataframe(scored.sort_values("churn_probability", ascending=False).head(25), width="stretch")
    st.download_button("Download all predictions (CSV)", scored.to_csv(index=False).encode(),
                       "churn_predictions.csv", "text/csv")

# ------------------------------------------------------------- comparison
with tab_cmp:
    if y is None:
        st.info("Upload data with a Churn column to compare models.")
    else:
        table = pd.DataFrame(
            {n: score_all(y, None, get_model(n).predict_proba(X)[:, 1], cutoff) for n in MODEL_PKL}
        ).T.round(4)
        st.dataframe(table.style.highlight_max(axis=0, color="#fde0c5"), width="stretch")
        pick = st.multiselect("Metrics to chart", list(table.columns), default=["AUC", "F1", "MCC"])
        if pick:
            st.bar_chart(table[pick])

# ---------------------------------------------------------- single customer
with tab_one:
    st.write("Describe one customer and get an individual churn probability from the selected model.")
    ref = raw_df.drop(columns=["customerID", "Churn"], errors="ignore")
    with st.form("one_customer"):
        a, b, c = st.columns(3)
        tenure = a.slider("Tenure (months)", 0, 72, 12)
        monthly = b.number_input("Monthly charges", 15.0, 130.0, 70.0, 0.5)
        contract = c.selectbox("Contract", sorted(ref["Contract"].unique()))
        internet = a.selectbox("Internet service", sorted(ref["InternetService"].unique()))
        pay = b.selectbox("Payment method", sorted(ref["PaymentMethod"].unique()))
        paperless = c.selectbox("Paperless billing", ["Yes", "No"])
        senior = a.selectbox("Senior citizen", ["No", "Yes"])
        partner = b.selectbox("Partner", ["No", "Yes"])
        deps = c.selectbox("Dependents", ["No", "Yes"])
        phone = a.selectbox("Phone service", ["Yes", "No"])
        multi = b.selectbox("Multiple lines", sorted(ref["MultipleLines"].unique()))
        addons = st.multiselect("Add-on services held",
                                ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                 "TechSupport", "StreamingTV", "StreamingMovies"])
        go = st.form_submit_button("Predict churn risk")
    if go:
        row = {
            "customerID": "manual", "gender": "Male", "SeniorCitizen": int(senior == "Yes"),
            "Partner": partner, "Dependents": deps, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi, "InternetService": internet,
            "PaperlessBilling": paperless, "Contract": contract, "PaymentMethod": pay,
            "MonthlyCharges": monthly, "TotalCharges": round(monthly * tenure, 2),
        }
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
            row[col] = "Yes" if col in addons else ("No internet service" if internet == "No" else "No")
        Xo, _ = prepare_telco_frame(pd.DataFrame([row]))
        p = float(model.predict_proba(Xo)[0, 1])
        st.metric("Churn probability", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.write("**Verdict:** likely to churn" if p >= cutoff else "**Verdict:** likely to stay")

# ----------------------------------------------------- feature importance
with tab_imp:
    if y is None:
        st.info("Upload data with a Churn column to see feature importance.")
    else:
        st.write("Permutation importance: how much the model's AUC drops when one feature is shuffled.")
        imp = importance_table(chosen, raw_df.to_json())
        fig, ax = plt.subplots(figsize=(6, 5))
        top = imp.head(12).iloc[::-1]
        ax.barh(top["feature"], top["auc_drop"], color="#d95f02")
        ax.set_xlabel("Drop in AUC")
        st.pyplot(fig)

# ---------------------------------------------------------- cut-off explorer
with tab_thr:
    if y is None:
        st.info("Upload data with a Churn column to explore cut-offs.")
    else:
        grid = np.arange(0.05, 0.96, 0.05)
        prec = [precision_score(y, proba >= t, zero_division=0) for t in grid]
        rec = [recall_score(y, proba >= t, zero_division=0) for t in grid]
        f1s = [f1_score(y, proba >= t, zero_division=0) for t in grid]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(grid, prec, label="Precision"); ax.plot(grid, rec, label="Recall"); ax.plot(grid, f1s, label="F1")
        ax.axvline(cutoff, color="grey", ls="--")
        ax.set_xlabel("Cut-off"); ax.legend()
        st.pyplot(fig)
        st.caption("Lower cut-offs catch more churners (recall up) at the price of more false alarms (precision down).")
