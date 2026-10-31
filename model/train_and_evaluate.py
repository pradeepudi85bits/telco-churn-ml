"""Tune, train, evaluate and save the five churn classifiers.

Run from the project root:  python model/train_and_evaluate.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
from preprocessing import prepare_telco_frame, make_preprocessor

SEED = 1337          # <- change to your own number, then re-run
TEST_FRACTION = 0.25
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw = pd.read_csv(os.path.join(ROOT, "data_full.csv"))
X, y = prepare_telco_frame(raw)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)

# hold-out set in raw format (incl. Churn) - this is what the Streamlit app uploads
raw.loc[X_te.index].to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

# model + search space (the pipeline step is called 'clf')
search_plan = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [10, 25, 50]}),
    "kNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [7, 11, 15, 21, 31], "clf__weights": ["uniform", "distance"]}),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}),
    "Support Vector Machine": (
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]}),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, best_params = [], {}

for name, (estimator, grid) in search_plan.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    if name == "Support Vector Machine":      # need probabilities for AUC / app
        best.set_params(clf__probability=True)
        best.fit(X_tr, y_tr)
    best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    best_params[name]["cv_f1"] = round(float(gs.best_score_), 4)

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_te, pred), "AUC": roc_auc_score(y_te, proba),
        "Precision": precision_score(y_te, pred), "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred), "MCC": matthews_corrcoef(y_te, pred),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    })
    joblib.dump(best, os.path.join(ROOT, "model", name.lower().replace(" ", "_") + ".pkl"))
    print(f"done: {name}  {best_params[name]}")

summary = pd.DataFrame(rows).round(4)
summary.to_csv(os.path.join(ROOT, "model", "metrics_summary.csv"), index=False)
with open(os.path.join(ROOT, "model", "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2, default=str)

print("\n", summary.to_string(index=False))
print("\ntrain:", X_tr.shape, " test:", X_te.shape)
