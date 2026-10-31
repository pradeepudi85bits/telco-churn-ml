# Churn Radar - Telco Customer Churn Prediction (ML Assignment 2, M.Tech DSE, BITS Pilani)

## a. Problem statement
A telecom operator loses recurring revenue every time a customer cancels. Using customer account, service and billing
details, this project trains and compares five classification models (Logistic Regression, Decision Tree, kNN,
Gaussian Naive Bayes, SVM) to predict whether a customer will churn, and exposes them through an interactive Streamlit app
("Churn Radar") that scores customers and lists the highest-risk ones.

## b. Dataset description
- **Source:** Kaggle - *Telco Customer Churn* (IBM sample data), `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Task:** binary classification, target `Churn` (Yes = 1, No = 0).
- **Size:** 7,043 customers; 20 raw input columns (+ `customerID`, dropped, + target) - above the minimum of 12 features / 500 rows.
- **Class balance:** 73.5 % stay / 26.5 % churn (imbalanced).
- **Raw features:** gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
  MonthlyCharges, TotalCharges.
- **Preparation (see `preprocessing.py`):**
  - `TotalCharges` is text in the CSV; converted to numeric, 11 blanks (tenure = 0 customers) set to 0.
  - `gender` removed (no meaningful signal); `customerID` removed.
  - Two engineered features: `avg_monthly_spend = TotalCharges / (tenure + 1)` and `service_count` (number of add-on services held).
  - Numeric columns standardised; categorical columns one-hot encoded - all inside a scikit-learn `Pipeline` per model.
- **Split:** 75 / 25 stratified, `random_state = 1337` -> 5,282 train / 1,761 test rows. The test split is saved as `test_data.csv`.
- **Tuning:** each model tuned with 5-fold stratified `GridSearchCV` (scoring = F1 on the churn class). `class_weight="balanced"`
  is used for Logistic Regression, Decision Tree and SVM to counter the class imbalance.

## c. GitHub repository link
https://github.com/pradeepudi85bits/telco-churn-m
Live Streamlit app: 
https://telco-churn-ml-appgxofkccezrageu4mm6gb.streamlit.app/?utm_source=chatgpt.com

```
telco-churn-ml/
|-- app.py                     # Streamlit app (Churn Radar)
|-- preprocessing.py           # cleaning, feature engineering, preprocessing pipeline
|-- requirements.txt
|-- README.md
|-- test_data.csv              # 1,761-row hold-out set (raw format, with Churn)
|-- data_full.csv              # original dataset used for training
|-- model/
    |-- train_and_evaluate.py  # tuning, training, evaluation, saving
    |-- metrics_summary.csv    # metrics + confusion-matrix counts
    |-- best_params.json       # best hyper-parameters per model
    |-- logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, support_vector_machine.pkl
```

Run locally:
```
pip install -r requirements.txt
python model/train_and_evaluate.py   # optional: re-tune, retrain, regenerate pkl files + test_data.csv
streamlit run app.py
```

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7553 | 0.8507 | 0.5253 | 0.8009 | 0.6344 | 0.4854 |
| Decision Tree | 0.7518 | 0.8184 | 0.5229 | 0.7323 | 0.6102 | 0.4488 |
| kNN | 0.8041 | 0.8359 | 0.6525 | 0.5589 | 0.6021 | 0.4757 |
| Naive Bayes | 0.7053 | 0.8262 | 0.4688 | 0.8373 | 0.6011 | 0.4376 |
| Support Vector Machine | 0.7967 | 0.8244 | 0.6172 | 0.6146 | 0.6159 | 0.4777 |

Precision, Recall and F1 are for the positive class (Churn = Yes) on the 1,761-row test set at a 0.5 probability cut-off.

Best hyper-parameters found by GridSearchCV:

| Model | Best parameters | CV F1 |
|---|---|---|
| Logistic Regression | C = 1 | 0.6297 |
| Decision Tree | max_depth = 6, min_samples_leaf = 50 | 0.6333 |
| kNN | n_neighbors = 31, weights = uniform | 0.5807 |
| Naive Bayes | var_smoothing = 0.001 | 0.5968 |
| SVM (RBF) | C = 1, gamma = scale | 0.6203 |

Confusion-matrix counts on the test set (TN / FP / FN / TP): Logistic Regression 956 / 338 / 93 / 374,
Decision Tree 982 / 312 / 125 / 342, kNN 1155 / 139 / 206 / 261, Naive Bayes 851 / 443 / 76 / 391,
SVM 1116 / 178 / 180 / 287.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.851), F1 (0.634) and MCC (0.485). Recall is high (0.80, catches 374 of 467 churners) but precision is only 0.53 because class weighting pushes it to flag many customers, which lowers accuracy to 0.755. |
| Decision Tree | AUC 0.818, F1 0.610, MCC 0.449. Recall 0.73 and precision 0.52, so it behaves like Logistic Regression with slightly weaker ranking quality (lower AUC). |
| kNN | Highest accuracy (0.804) and precision for the non-SVM models (0.653), but recall is low (0.559): it misses 206 of 467 churners. Accuracy looks best mostly because it predicts "stay" more often. |
| Naive Bayes | Highest recall (0.837, only 76 churners missed) but lowest precision (0.469) and accuracy (0.705); 443 false alarms. Lowest MCC (0.438). |
| Support Vector Machine | Balanced precision (0.617) and recall (0.615), F1 0.616 (second best) and accuracy 0.797, but AUC (0.824) is below Logistic Regression and kNN. |
| **Overall winner for my dataset** | **Logistic Regression** - best AUC, F1 and MCC (the metrics that matter most with an imbalanced target). If plain accuracy or precision matters more, kNN or SVM are the alternatives. |

## Streamlit app features
- CSV upload of test data (falls back to the bundled `test_data.csv`)
- Model selection dropdown (5 models) and an adjustable churn-probability cut-off
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix, classification report and ROC curve
- Model comparison tab (all 5 models side by side, chartable)
- Single-customer predictor form
- Permutation feature-importance chart
- Cut-off explorer (precision / recall / F1 vs cut-off)
- Table of highest-risk customers and downloadable predictions

## Deployment
The machine learning application was deployed using Streamlit Community Cloud. The application is connected to my GitHub repository and can be accessed through the link below.

Live Application:
https://telco-churn-ml-appgxofkccezrageu4mm6gb.streamlit.app/
