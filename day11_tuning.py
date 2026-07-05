# ==========================================
# DAY 11 - HYPERPARAMETER TUNING
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load Data

df = pd.read_csv(
    "customer_churn_data.csv"
)

X = df[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = df["Churn"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = XGBClassifier(
    random_state=42
)

# Parameter Grid

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1]
}

# Grid Search

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")

print(
    grid.best_params_
)

best_model = grid.best_estimator_

# Prediction

y_pred = best_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nTuned Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print("\nDAY 11 COMPLETED")