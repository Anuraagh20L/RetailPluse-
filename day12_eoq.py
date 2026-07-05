import pandas as pd
import numpy as np

annual_demand = 30838 * 365
ordering_cost = 500
holding_cost = 50

eoq = np.sqrt(
    (2 * annual_demand * ordering_cost)
    / holding_cost
)

print("EOQ:", round(eoq, 2))

result = pd.DataFrame({
    "Metric": ["EOQ"],
    "Value": [round(eoq, 2)]
})

result.to_csv(
    "eoq_results.csv",
    index=False
)

print("eoq_results.csv saved successfully")
print("DAY 12 COMPLETED")