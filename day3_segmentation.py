# ==========================================
# DAY 3 - CUSTOMER SEGMENTATION
# K-MEANS + DBSCAN CLUSTERING
# ==========================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ==========================================
# LOAD RFM DATA
# ==========================================

rfm = pd.read_csv("rfm_data.csv")

print("Dataset Shape:")
print(rfm.shape)

print("\nFirst 5 Rows")
print(rfm.head())

# ==========================================
# SELECT FEATURES
# ==========================================

rfm_features = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm_features
)

print("\nScaled Data")
print(rfm_scaled[:5])

# ==========================================
# ELBOW METHOD
# ==========================================

inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    inertia.append(
        model.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    inertia,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid(True)

plt.show()

# ==========================================
# K-MEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

print("\nCluster Counts")
print(
    rfm["Cluster"]
    .value_counts()
)

# ==========================================
# SILHOUETTE SCORE
# ==========================================

score = silhouette_score(
    rfm_scaled,
    rfm["Cluster"]
)

print(
    "\nSilhouette Score:",
    round(score,3)
)

# ==========================================
# CLUSTER SUMMARY
# ==========================================

cluster_summary = rfm.groupby(
    "Cluster"
)[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
].mean()

print("\nCluster Summary")
print(cluster_summary)

# ==========================================
# CUSTOMER SEGMENT LABELS
# ==========================================

segment_map = {
    0: "VIP Customers",
    1: "Loyal Customers",
    2: "At Risk Customers",
    3: "New Customers"
}

rfm["Segment"] = (
    rfm["Cluster"]
    .map(segment_map)
)

print("\nSegment Counts")

print(
    rfm["Segment"]
    .value_counts()
)

# ==========================================
# K-MEANS VISUALIZATION
# ==========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="Segment",
    palette="Set1"
)

plt.title(
    "Customer Segmentation using K-Means"
)

plt.show()

# ==========================================
# FREQUENCY VS MONETARY
# ==========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Segment",
    palette="Set2"
)

plt.title(
    "Frequency vs Monetary"
)

plt.show()

# ==========================================
# DBSCAN CLUSTERING
# ==========================================

dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

rfm["DBSCAN_Cluster"] = (
    dbscan.fit_predict(
        rfm_scaled
    )
)

print("\nDBSCAN Clusters")

print(
    rfm["DBSCAN_Cluster"]
    .value_counts()
)

# ==========================================
# DBSCAN VISUALIZATION
# ==========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="DBSCAN_Cluster",
    palette="Set3"
)

plt.title(
    "DBSCAN Customer Segmentation"
)

plt.show()

# ==========================================
# TOP CUSTOMERS
# ==========================================

top_customers = rfm.sort_values(
    by="Monetary",
    ascending=False
).head(10)

print("\nTop 10 Customers")

print(
    top_customers[
        [
            "CustomerID",
            "Monetary",
            "Frequency",
            "Recency"
        ]
    ]
)

# ==========================================
# SAVE RESULTS
# ==========================================

rfm.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nDay 3 Completed Successfully")

print(
    "\nFile Saved: customer_segments.csv"
)