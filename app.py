import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

div[data-testid="stMetric"]{
    background:#1E293B;
    padding:18px;
    border-radius:15px;
    border:1px solid #3B82F6;
}

h1,h2,h3{
    color:#4FC3F7;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 RetailPulse Dashboard")
st.markdown("### AI-Powered Customer Analytics & Demand Forecasting Platform")

# Sidebar
page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Customer Analytics",
        "Demand Forecasting",
        "Inventory Optimization",
        "Real-Time Monitoring",
        "Churn Prediction"
    ]
)

# Executive Overview
if page == "Executive Overview":

    st.header("📊 Executive Overview")

    # Load datasets
    customers = pd.read_csv("customer_segments.csv")
    forecast = pd.read_csv("hybrid_forecast.csv")
    inventory = pd.read_csv("inventory_recommendations.csv")
    churn = pd.read_csv("customer_churn_data.csv")

    # KPIs
    total_customers = customers["CustomerID"].nunique()
    avg_forecast = forecast["HybridForecast"].mean()
    recommended_inventory = inventory.loc[
        inventory["Metric"] == "Recommended Inventory",
        "Value"
    ].values[0]
    churn_rate = (churn["Churn"].sum() / len(churn)) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Customers", total_customers)
    col2.metric("📈 Avg Forecast", f"{avg_forecast:,.2f}")
    col3.metric("📦 Recommended Inventory", f"{recommended_inventory:,.0f}")
    col4.metric("⚠ Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    st.subheader("Customer Segments")

    st.bar_chart(
        customers["Segment"].value_counts()
    )

    st.divider()

    st.subheader("Demand Forecast")

    st.line_chart(
        forecast.set_index("Date")["HybridForecast"]
    )

    st.divider()

    st.subheader("Inventory Overview")

    st.bar_chart(
        inventory.set_index("Metric")
    )

# Customer Analytics
elif page == "Customer Analytics":

    st.header("👥 Customer Analytics Dashboard")

    # Load customer data
    customers = pd.read_csv("customer_segments.csv")

    # KPIs
    total_customers = customers["CustomerID"].nunique()
    vip = (customers["Segment"] == "VIP Customers").sum()
    loyal = (customers["Segment"] == "Loyal Customers").sum()
    at_risk = (customers["Segment"] == "At Risk Customers").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("VIP Customers", vip)
    col3.metric("Loyal Customers", loyal)
    col4.metric("At Risk", at_risk)

    st.divider()

    # Customer Segment Distribution
    st.subheader("Customer Segment Distribution")

    segment_count = customers["Segment"].value_counts()

    st.bar_chart(segment_count)

    st.divider()

    # Top Customers
    st.subheader("Top 10 Customers by Monetary Value")

    top10 = customers.sort_values(
        "Monetary",
        ascending=False
    ).head(10)

    st.dataframe(
        top10[
            ["CustomerID", "Monetary", "Segment"]
        ]
    )

    st.divider()

    # Segment-wise statistics
    st.subheader("Average Monetary Value by Segment")

    avg_value = customers.groupby("Segment")["Monetary"].mean()

    st.bar_chart(avg_value)

# Demand Forecasting
elif page == "Demand Forecasting":

    st.header("Demand Forecasting Dashboard")

    forecast = pd.read_csv("hybrid_forecast.csv")

    st.subheader("Forecast Data")
    st.dataframe(forecast)

    st.subheader("Hybrid Forecast Trend")
    st.line_chart(
        forecast.set_index("Date")["HybridForecast"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Forecast",
        f"{forecast['HybridForecast'].mean():,.2f}"
    )

    col2.metric(
        "Maximum Forecast",
        f"{forecast['HybridForecast'].max():,.2f}"
    )

    col3.metric(
        "Minimum Forecast",
        f"{forecast['HybridForecast'].min():,.2f}"
    )

    st.divider()

    st.subheader("What-If Analysis")

    growth = st.slider(
        "Expected Demand Increase (%)",
        0,
        50,
        10
    )

    forecast["Adjusted Forecast"] = (
        forecast["HybridForecast"] * (1 + growth / 100)
    )

    st.line_chart(
        forecast.set_index("Date")[
            ["HybridForecast", "Adjusted Forecast"]
        ]
    )

    st.dataframe(
        forecast[
            ["Date", "HybridForecast", "Adjusted Forecast"]
        ]
    )

# Inventory
elif page == "Inventory Optimization":

    st.header("📦 Inventory Optimization Dashboard")

    # Load inventory data
    inventory = pd.read_csv("inventory_recommendations.csv")
    eoq = pd.read_csv("eoq_results.csv")

    st.subheader("Inventory Metrics")
    st.dataframe(inventory)

    st.divider()

    # KPI Cards
    metrics = inventory.set_index("Metric")["Value"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Recommended Inventory",
        f"{metrics['Recommended Inventory']:,.2f}"
    )

    col2.metric(
        "Reorder Point",
        f"{metrics['Reorder Point']:,.2f}"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Safety Stock",
        f"{metrics['Safety Stock']:,.2f}"
    )

    col4.metric(
        "Average Demand",
        f"{metrics['Average Demand']:,.2f}"
    )

    st.divider()

    st.subheader("Inventory Metrics Chart")

    st.bar_chart(
        inventory.set_index("Metric")
    )

    st.divider()

    st.subheader("Economic Order Quantity (EOQ)")

    st.metric(
    "EOQ",
    f"{eoq['Value'].iloc[0]:,.2f}"
)
    
elif page == "Real-Time Monitoring":

    st.header("📡 Real-Time Metrics & Alerts")

    # Load datasets
    forecast = pd.read_csv("hybrid_forecast.csv")
    inventory = pd.read_csv("inventory_recommendations.csv")

    # Live KPIs
    avg_forecast = forecast["HybridForecast"].mean()
    max_forecast = forecast["HybridForecast"].max()
    inventory_level = inventory.loc[
        inventory["Metric"] == "Recommended Inventory",
        "Value"
    ].values[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Forecast",
        f"{avg_forecast:,.2f}"
    )

    col2.metric(
        "Peak Forecast",
        f"{max_forecast:,.2f}"
    )

    col3.metric(
        "Inventory Level",
        f"{inventory_level:,.0f}"
    )

    st.divider()

    st.subheader("Forecast Trend")

    st.line_chart(
        forecast.set_index("Date")["HybridForecast"]
    )

    st.divider()

    st.subheader("System Alerts")

    if avg_forecast > 30000:
        st.warning("⚠ High demand detected. Consider increasing inventory.")

    if inventory_level < 200000:
        st.error("🚨 Inventory is below recommended level.")

    else:
        st.success("✅ Inventory level is healthy.")

    st.divider()

    st.subheader("Forecast Data")

    st.dataframe(forecast)
    
# Churn
elif page == "Churn Prediction":

    st.header("⚠ Customer Churn Dashboard")

    churn = pd.read_csv("customer_churn_data.csv")

    st.subheader("Customer Churn Dataset")

    st.dataframe(churn.head())

    st.divider()

    # KPIs
    total = len(churn)
    churned = (churn["Churn"] == 1).sum()
    active = (churn["Churn"] == 0).sum()
    churn_rate = (churned / total) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total)
    col2.metric("Active Customers", active)
    col3.metric("Churn Customers", churned)
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    st.subheader("Churn Distribution")

    st.bar_chart(
        churn["Churn"].value_counts()
    )

    st.divider()

    st.subheader("Recency vs Churn")

    st.scatter_chart(
        churn[["Recency", "Churn"]]
    )

    st.divider()

    st.subheader("Feature Summary")

    st.write(
        churn[
            ["Recency", "Frequency", "Monetary"]
        ].describe()
    )
# ==============================
# DAY 20 - EXPORT REPORTS
# ==============================

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Reports")

files = {
    "Customer Segments": "customer_segments.csv",
    "Hybrid Forecast": "hybrid_forecast.csv",
    "Customer Churn": "customer_churn_data.csv",
    "Inventory": "inventory_recommendations.csv",
    "EOQ Results": "eoq_results.csv"
}

selected = st.sidebar.selectbox(
    "Choose Report",
    list(files.keys())
)

with open(files[selected], "rb") as file:
    st.sidebar.download_button(
        label="⬇ Download CSV",
        data=file,
        file_name=files[selected],
        mime="text/csv"
    )    