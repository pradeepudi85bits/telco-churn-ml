"""Train the 5 classifiers on Telco Customer Churn, evaluate, and save artefacts."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from utils import clean_data, build_preprocessor

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = pd.read_csv(os.path.join(ROOT, "data_full.csv"))
X, y = clean_data(raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Save raw-format test data (with Churn) for the Streamlit app
test_raw = raw.loc[X_test.index]
test_raw.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
}

rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)  # same rule the app uses at default threshold
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipe, os.path.join(ROOT, "model", fname))

res = pd.DataFrame(rows).round(4)
res.to_csv(os.path.join(ROOT, "model", "metrics_summary.csv"), index=False)
print(res.to_string(index=False))
print("\nTrain:", X_train.shape, "Test:", X_test.shape)
