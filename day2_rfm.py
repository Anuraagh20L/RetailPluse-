# ============================
# DAY 2 - RFM ANALYSIS
# ============================

import pandas as pd
import matplotlib.pyplot as plt

# Load Cleaned Data
df = pd.read_csv(
    "cleaned_retail_data.csv"
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

# -------------------------
# Date Features
# -------------------------

df["Year"] = (
    df["InvoiceDate"]
    .dt.year
)

df["Month"] = (
    df["InvoiceDate"]
    .dt.month
)

df["Day"] = (
    df["InvoiceDate"]
    .dt.day
)

df["Weekday"] = (
    df["InvoiceDate"]
    .dt.day_name()
)

# -------------------------
# RFM Table
# -------------------------

snapshot_date = (
    df["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

rfm = df.groupby(
    "CustomerID"
).agg({
    "InvoiceDate":
        lambda x:
        (
            snapshot_date
            - x.max()
        ).days,

    "InvoiceNo":
        "count",

    "TotalPrice":
        "sum"
})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print(rfm.head())

# -------------------------
# RFM Scores
# -------------------------

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5,4,3,2,1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"]
    .rank(
        method="first"
    ),
    5,
    labels=[1,2,3,4,5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1,2,3,4,5]
)

rfm["RFM_Score"] = (
    rfm["R_Score"]
    .astype(str)
    +
    rfm["F_Score"]
    .astype(str)
    +
    rfm["M_Score"]
    .astype(str)
)

# -------------------------
# Customer Categories
# -------------------------

rfm["CustomerType"] = "Regular"

rfm.loc[
    rfm["RFM_Score"] == "555",
    "CustomerType"
] = "VIP Customer"

rfm.loc[
    rfm["R_Score"].astype(int) >= 4,
    "CustomerType"
] = "Loyal Customer"

print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "RFM_Score",
            "CustomerType"
        ]
    ].head()
)

# -------------------------
# Visualizations
# -------------------------

plt.figure(figsize=(8,5))

plt.hist(
    rfm["Recency"],
    bins=30
)

plt.title(
    "Recency Distribution"
)

plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    rfm["Frequency"],
    bins=30
)

plt.title(
    "Frequency Distribution"
)

plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    rfm["Monetary"],
    bins=30
)

plt.title(
    "Monetary Distribution"
)

plt.show()

# Save RFM Dataset
rfm.to_csv(
    "rfm_data.csv"
)

print("Day 2 Completed")