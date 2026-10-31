# Telco Customer Churn Prediction - ML Assignment 2 (M.Tech DSE, BITS Pilani)

## a. Problem statement
A telecom company loses revenue whenever a customer leaves (churns). The goal is to build and compare five
classification models that predict, from a customer's account and service details, whether the customer will
churn (`Yes`) or stay (`No`). The best model can help the business target retention offers at high-risk customers.
The models are exposed through an interactive Streamlit web app.

## b. Dataset description
- **Source:** Kaggle - *Telco Customer Churn* (IBM sample data), file `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Task:** Binary classification (target = `Churn`).
- **Size:** 7,043 instances, 20 input features (+ `customerID`, dropped, + target) - satisfies the minimum of 12 features / 500 instances.
- **Class balance:** No = 5,174 (73.5 %), Yes = 1,869 (26.5 %) -> imbalanced, so a stratified split is used.
- **Features:**
  - *Demographics:* gender, SeniorCitizen, Partner, Dependents
  - *Account:* tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
  - *Services:* PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Preprocessing:** `customerID` dropped; `TotalCharges` converted from text to numeric (11 blanks for tenure-0 customers set to 0);
  numeric columns standardised (`StandardScaler`); categorical columns one-hot encoded; all inside one scikit-learn `Pipeline`
  per model so the Streamlit app can accept raw CSV rows.
- **Split:** 80 / 20 stratified train-test split, `random_state=42` (5,634 train / 1,409 test). The test split is saved as `test_data.csv`.

## c. GitHub repository link
`<PASTE YOUR GITHUB REPO URL HERE>`

Live Streamlit app: `<PASTE YOUR STREAMLIT URL HERE>`

```
telco-churn-ml/
|-- app.py                 # Streamlit application
|-- utils.py               # cleaning + preprocessing shared by training and app
|-- requirements.txt
|-- README.md
|-- test_data.csv          # 1,409-row hold-out test set (with Churn column)
|-- data_full.csv          # original dataset (used by the training script)
|-- model/
    |-- train_models.py    # trains / evaluates / saves all 5 models
    |-- metrics_summary.csv
    |-- logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, support_vector_machine.pkl
```

Run locally:
```
pip install -r requirements.txt
python model/train_models.py      # optional - re-trains and regenerates pkl files + test_data.csv
streamlit run app.py
```

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8420 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7984 | 0.8276 | 0.6347 | 0.5668 | 0.5989 | 0.4662 |
| kNN | 0.7814 | 0.8225 | 0.5943 | 0.5561 | 0.5746 | 0.4281 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | 0.8396 | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7928 | 0.7928 | 0.6614 | 0.4492 | 0.5350 | 0.4205 |

Metrics are for the positive class (Churn = Yes) on the 1,409-row test set, default 0.5 probability threshold.
Hyper-parameters: LR (`max_iter=1000`), Decision Tree (`max_depth=5, min_samples_leaf=20`), kNN (`k=15`),
GaussianNB (defaults), SVM (RBF kernel, `C=1.0`, `probability=True`).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy, AUC, F1 and MCC. Churn is largely a linear function of tenure, contract type and charges, so a simple, well-regularised linear model fits well and gives well-calibrated probabilities. |
| Decision Tree | Close second (accuracy 0.798, highest recall among the non-NB models at 0.567). Depth was limited to 5 to avoid overfitting; it is very interpretable (contract type is the root split, followed by internet service and monthly charges) but its AUC is slightly lower than LR because of step-wise probabilities. |
| kNN | Weakest of the distance/tree/linear group on precision and F1. With ~40 one-hot features after encoding, distances become less informative (curse of dimensionality), so neighbours are noisier; it is also the slowest at prediction time. |
| Naive Bayes | Lowest accuracy (0.696) and precision (0.460) but by far the highest recall (0.840). The independence assumption is violated by correlated service columns, producing over-confident churn predictions - it catches most churners but raises many false alarms. Useful only if missing a churner is much costlier than a false alert. |
| Support Vector Machine | Accuracy is competitive (0.793) and precision is high (0.661), but recall is the lowest (0.449), so it misses over half of churners because it favours the majority class. Lowest AUC (0.793) and MCC; class weighting or tuning of C/gamma would likely help. |
| **Overall winner for this dataset** | **Logistic Regression** - best Accuracy, AUC, F1 and MCC while remaining simple, fast and interpretable. (If recall on churners is the priority, Naive Bayes or a lower decision threshold on LR is the alternative.) |

## Streamlit app features
1. CSV upload of test data (falls back to bundled `test_data.csv`)
2. Model selection dropdown (5 models)
3. Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
4. Confusion matrix, classification report and ROC curve
5. Extras: adjustable decision threshold, all-model comparison table/chart, downloadable predictions

## Deployment (Streamlit Community Cloud)
Push the repo to GitHub -> https://streamlit.io/cloud -> *New app* -> select repo, branch `main`, file `app.py` -> *Deploy*.
