# ==========================================
# DAY 9 - CUSTOMER CHURN PREDICTION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

from xgboost import XGBClassifier

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "customer_churn_data.csv"
)

print("Dataset Shape:")
print(df.shape)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = df["Churn"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# ==========================================
# ROC CURVE
# ==========================================

y_prob = model.predict_proba(
    X_test
)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.legend()

plt.title(
    "ROC Curve"
)

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance"
)

plt.show()

# ==========================================
# SAVE PREDICTIONS
# ==========================================

pred_df = X_test.copy()

pred_df["Actual"] = y_test.values
pred_df["Predicted"] = y_pred

pred_df.to_csv(
    "churn_predictions.csv",
    index=False
)

print(
    "\nchurn_predictions.csv saved successfully"
)

print(
    "\nDAY 9 COMPLETED"
)