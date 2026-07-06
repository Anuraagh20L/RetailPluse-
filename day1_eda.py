
# DAY 1 - EDA + DATA CLEANING
# RETAILPULSE PROJECT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATASET

file_path = r"ExtractedData\online_retail_II.xlsx"

df1 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df2 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

# Combine both years

df = pd.concat(
    [df1, df2],
    ignore_index=True
)


# BASIC INFORMATION

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# RENAME COLUMNS

df.rename(columns={
    "Invoice": "InvoiceNo",
    "Customer ID": "CustomerID",
    "Price": "UnitPrice"
}, inplace=True)


# DATA CLEANING

print("\nCleaning Data...")

# Remove missing Customer IDs
df.dropna(
    subset=["CustomerID"],
    inplace=True
)

# Remove duplicates
df.drop_duplicates(
    inplace=True
)

# Remove cancelled invoices
df = df[
    ~df["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
]

# Remove negative quantities
df = df[
    df["Quantity"] > 0
]

# Remove invalid prices
df = df[
    df["UnitPrice"] > 0
]

# Create TotalPrice column
df["TotalPrice"] = (
    df["Quantity"]
    * df["UnitPrice"]
)

# Convert date column
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

df["StockCode"]=df["StockCode"].astype(str).str.strip()

print("\nCleaned Dataset Shape:")
print(df.shape)


# TOP 10 COUNTRIES BY SALES

country_sales = (
    df.groupby("Country")
    ["TotalPrice"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(10,5))

country_sales.plot(
    kind="bar"
)

plt.title(
    "Top 10 Countries by Sales"
)

plt.xlabel("Country")
plt.ylabel("Sales")

plt.tight_layout()

plt.show()


# TOP 10 PRODUCTS
top_products = (
    df.groupby("Description")
    ["Quantity"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12,5))

top_products.plot(
    kind="bar"
)

plt.title(
    "Top 10 Selling Products"
)

plt.xlabel("Product")
plt.ylabel("Quantity Sold")

plt.tight_layout()

plt.show()

# MONTHLY SALES TREND

df["Month"] = (
    df["InvoiceDate"]
    .dt.to_period("M")
)

monthly_sales = (
    df.groupby("Month")
    ["TotalPrice"]
    .sum()
)

plt.figure(figsize=(12,5))

monthly_sales.plot()

plt.title(
    "Monthly Sales Trend"
)

plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.tight_layout()

plt.show()


# OUTLIER ANALYSIS

# Quantity Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Quantity"])
plt.title("Quantity Outliers")
plt.show()

# Unit Price Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df["UnitPrice"])
plt.title("Unit Price Outliers")
plt.show()

# TotalPrice Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df["TotalPrice"])
plt.title("TotalPrice Outliers")
plt.show()
#plt.figure(figsize=(8,5))


import matplotlib.pyplot as plt
import seaborn as sns

# Set up a 1x3 grid for the subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Quantity Boxplot
sns.boxplot(x=df["Quantity"], ax=axes[0], color="skyblue")
axes[0].set_title("Quantity Outliers")

# Unit Price Boxplot
sns.boxplot(x=df["UnitPrice"], ax=axes[1], color="salmon")
axes[1].set_title("Unit Price Outliers")

# Total Price Boxplot
sns.boxplot(x=df["TotalPrice"], ax=axes[2], color="lightgreen")
axes[2].set_title("TotalPrice Outliers")

# Adjust layout so titles and labels don't overlap
plt.tight_layout()
plt.show()


# CORRELATION HEATMAP
sns.heatmap(
    df[
        [
            "Quantity",
            "UnitPrice",
            "TotalPrice"
        ]
    ].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.show()

# SAVE CLEANED DATASET
df.to_csv(
    "cleaned_retail_data.csv",
    index=False
)

print("\nFile Saved:")
print("cleaned_retail_data.csv")

print("\nDAY 1 COMPLETED SUCCESSFULLY")
