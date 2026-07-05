import pandas as pd

df = pd.read_csv("hybrid_forecast.csv")

avg_demand = df["HybridForecast"].mean()
std_demand = df["HybridForecast"].std()

lead_time = 7
service_factor = 1.65

safety_stock = service_factor * std_demand

reorder_point = (
    avg_demand * lead_time
) + safety_stock

recommended_inventory = (
    avg_demand * 30
) + safety_stock

results = pd.DataFrame({
    "Metric":[
        "Average Demand",
        "Safety Stock",
        "Reorder Point",
        "Recommended Inventory"
    ],
    "Value":[
        avg_demand,
        safety_stock,
        reorder_point,
        recommended_inventory
    ]
})

print(results)

results.to_csv(
    "inventory_recommendations.csv",
    index=False
)

print(
    "\ninventory_recommendations.csv saved successfully"
)

print(
    "\nDAY 10 COMPLETED"
)