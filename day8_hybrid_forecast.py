# ==========================================
# DAY 8 - HYBRID FORECASTING
# PROPHET + LSTM ENSEMBLE
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD FILES
# ==========================================

prophet = pd.read_csv(
    "sales_forecast.csv"
)

lstm = pd.read_csv(
    "lstm_forecast.csv"
)

print("Prophet Shape:", prophet.shape)
print("LSTM Shape:", lstm.shape)

# ==========================================
# PREPARE DATA
# ==========================================

prophet = prophet.tail(30).reset_index(
    drop=True
)

lstm = lstm.reset_index(
    drop=True
)

# ==========================================
# HYBRID FORECAST
# ==========================================

hybrid = pd.DataFrame()

hybrid["Date"] = prophet["ds"]

hybrid["Prophet"] = prophet["yhat"]

hybrid["LSTM"] = lstm["Forecast"]

# Average Forecast

hybrid["HybridForecast"] = (
    hybrid["Prophet"]
    +
    hybrid["LSTM"]
) / 2

print("\nFirst 5 Rows")
print(hybrid.head())

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(
    hybrid["Date"],
    hybrid["Prophet"],
    label="Prophet"
)

plt.plot(
    hybrid["Date"],
    hybrid["LSTM"],
    label="LSTM"
)

plt.plot(
    hybrid["Date"],
    hybrid["HybridForecast"],
    linewidth=3,
    label="Hybrid Forecast"
)

plt.legend()

plt.title(
    "Hybrid Forecasting Model"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ==========================================
# SAVE FILE
# ==========================================

hybrid.to_csv(
    "hybrid_forecast.csv",
    index=False
)

print(
    "\nhybrid_forecast.csv saved successfully"
)

print(
    "\nDAY 8 COMPLETED"
)