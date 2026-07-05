# ==========================================
# DAY 4 - TIME SERIES PREPARATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# ==========================================
# LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("cleaned_retail_data.csv")

print("Dataset Loaded")

# ==========================================
# DATE CONVERSION
# ==========================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

# ==========================================
# DAILY SALES
# ==========================================

daily_sales = (
    df.groupby(
        df["InvoiceDate"].dt.date
    )["TotalPrice"]
    .sum()
    .reset_index()
)

daily_sales.columns = [
    "Date",
    "Sales"
]

print("\nFirst 5 Rows")
print(daily_sales.head())

# ==========================================
# DAILY SALES TREND
# ==========================================

plt.figure(figsize=(12,5))

plt.plot(
    daily_sales["Date"],
    daily_sales["Sales"]
)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()

# ==========================================
# DATE INDEX
# ==========================================

daily_sales["Date"] = pd.to_datetime(
    daily_sales["Date"]
)

daily_sales.set_index(
    "Date",
    inplace=True
)

# ==========================================
# ADF TEST
# ==========================================

result = adfuller(
    daily_sales["Sales"]
)

print("\nADF Statistic:")
print(result[0])

print("\np-value:")
print(result[1])

if result[1] < 0.05:
    print("\nData is Stationary")
else:
    print("\nData is NOT Stationary")

# ==========================================
# DECOMPOSITION
# ==========================================

decomposition = seasonal_decompose(
    daily_sales["Sales"],
    model="additive",
    period=30
)

# Trend

plt.figure(figsize=(12,5))
decomposition.trend.plot()
plt.title("Trend Component")
plt.show()

# Seasonal

plt.figure(figsize=(12,5))
decomposition.seasonal.plot()
plt.title("Seasonal Component")
plt.show()

# Residual

plt.figure(figsize=(12,5))
decomposition.resid.plot()
plt.title("Residual Component")
plt.show()

# Full Plot

fig = decomposition.plot()
fig.set_size_inches(12,8)

plt.show()

# ==========================================
# ROLLING AVERAGE
# ==========================================

daily_sales["RollingMean"] = (
    daily_sales["Sales"]
    .rolling(window=30)
    .mean()
)

plt.figure(figsize=(12,5))

plt.plot(
    daily_sales["Sales"],
    label="Actual Sales"
)

plt.plot(
    daily_sales["RollingMean"],
    label="30 Day Average"
)

plt.legend()

plt.title(
    "Sales vs Rolling Average"
)

plt.show()

# ==========================================
# SAVE FORECAST DATA
# ==========================================

forecast_df = daily_sales.reset_index()

forecast_df = forecast_df[
    ["Date","Sales"]
]

forecast_df.columns = [
    "ds",
    "y"
]

forecast_df.to_csv(
    "forecast_data.csv",
    index=False
)

print(
    "\nforecast_data.csv saved successfully"
)

print(
    "\nDAY 4 COMPLETED"
)