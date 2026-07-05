import pandas as pd

# Load Files

segments = pd.read_csv(
    "customer_segments.csv"
)

forecast = pd.read_csv(
    "hybrid_forecast.csv"
)

inventory = pd.read_csv(
    "inventory_recommendations.csv"
)

# -------------------------
# Customer Segments
# -------------------------

print("\nCustomer Segments")

print(
    segments["Segment"]
    .value_counts()
)

# -------------------------
# Top Customers
# -------------------------

top_customers = (
    segments
    .sort_values(
        by="Monetary",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Customers")

print(
    top_customers[
        [
            "CustomerID",
            "Monetary"
        ]
    ]
)

# -------------------------
# Forecast Summary
# -------------------------

print("\nForecast Summary")

print(
    forecast[
        "HybridForecast"
    ].describe()
)

# -------------------------
# Inventory Summary
# -------------------------

print("\nInventory Metrics")

print(inventory)

# -------------------------
# Save Dashboard Data
# -------------------------

dashboard_data = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Average Forecast"
    ],
    "Value": [
        len(segments),
        forecast[
            "HybridForecast"
        ].mean()
    ]
})

dashboard_data.to_csv(
    "dashboard_kpi.csv",
    index=False
)

print(
    "\ndashboard_kpi.csv saved successfully"
)

print(
    "\nDAY 13 COMPLETED"
)