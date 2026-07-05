# ==========================================
# DAY 5 - DEMAND FORECASTING WITH PROPHET
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# LOAD FORECAST DATA
# ==========================================

df = pd.read_csv("forecast_data.csv")

df["ds"] = pd.to_datetime(df["ds"])

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

print("\nTrain Shape:", train.shape)
print("Test Shape:", test.shape)

# ==========================================
# PROPHET MODEL
# ==========================================

model = Prophet()

model.fit(train)

# ==========================================
# FORECAST TEST PERIOD
# ==========================================

future = model.make_future_dataframe(
    periods=len(test)
)

forecast = model.predict(future)

# ==========================================
# EVALUATION
# ==========================================

predictions = forecast["yhat"].tail(
    len(test)
).values

actual = test["y"].values

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = (
    mean_squared_error(
        actual,
        predictions
    )
) ** 0.5

print("\nMAE:", round(mae,2))
print("RMSE:", round(rmse,2))

# ==========================================
# FORECAST PLOT
# ==========================================

fig1 = model.plot(forecast)

plt.title(
    "Prophet Sales Forecast"
)

plt.show()

# ==========================================
# COMPONENTS
# ==========================================

fig2 = model.plot_components(
    forecast
)

plt.show()

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(12,5))

plt.plot(
    test["ds"],
    actual,
    label="Actual"
)

plt.plot(
    test["ds"],
    predictions,
    label="Predicted"
)

plt.legend()

plt.title(
    "Actual vs Predicted Sales"
)

plt.show()

# ==========================================
# NEXT 30 DAYS FORECAST
# ==========================================

future_30 = model.make_future_dataframe(
    periods=30
)

forecast_30 = model.predict(
    future_30
)

forecast_30[
    [
        "ds",
        "yhat"
    ]
].tail(30).to_csv(
    "sales_forecast.csv",
    index=False
)

print(
    "\nsales_forecast.csv saved successfully"
)

print(
    "\nDAY 5 COMPLETED"
)