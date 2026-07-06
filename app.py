import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Business Revenue Forecasting",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# Load Model
# ==========================================

with open("models/revenue_model.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("outputs/featured_walmart.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ==========================================
# Title
# ==========================================

st.title("📈 Business Revenue Forecasting System")
st.write("Predict future revenue using Walmart historical sales data.")

st.divider()

# ==========================================
# User Inputs
# ==========================================

col1, col2 = st.columns(2)

with col1:
    store = st.number_input(
        "🏪 Store Number",
        min_value=1,
        step=1
    )

with col2:
    future_date = st.date_input(
        "📅 Future Prediction Date"
    )

# ==========================================
# Prediction Button
# ==========================================

if st.button("🚀 Predict Revenue", use_container_width=True):

    future_date = pd.to_datetime(future_date)

    # Date Validation
    if future_date < df["Date"].min():
        st.error(
            f"Dataset starts from {df['Date'].min().date()}"
        )
        st.stop()

    # Store Validation
    store_data = (
        df[df["Store"] == store]
        .sort_values("Date")
    )

    if store_data.empty:
        st.error("Invalid Store Number")
        st.stop()

    # Historical / Future Prediction
    if future_date <= df["Date"].max():
        latest = (
            store_data[
                store_data["Date"] <= future_date
            ].iloc[-1]
        )
    else:
        latest = store_data.iloc[-1]

    # Model Input
    input_data = pd.DataFrame({
        "Store":[store],
        "Holiday_Flag":[1 if future_date.month in [11,12] else 0],
        "Temperature":[latest["Temperature"]],
        "Fuel_Price":[latest["Fuel_Price"]],
        "CPI":[latest["CPI"]],
        "Unemployment":[latest["Unemployment"]],
        "Year":[future_date.year],
        "Month":[future_date.month],
        "Week":[future_date.isocalendar().week],
        "Quarter":[future_date.quarter],
        "Day":[future_date.day],
        "DayOfWeek":[future_date.dayofweek],
        "IsWeekend":[1 if future_date.dayofweek >= 5 else 0],
        "Previous_Week_Sales":[latest["Weekly_Sales"]],
        "Rolling_Mean_4":[latest["Rolling_Mean_4"]],
        "Rolling_STD_4":[latest["Rolling_STD_4"]],
        "Sales_Difference":[latest["Sales_Difference"]],
        "Revenue_Growth":[latest["Revenue_Growth"]]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"## 💰 Predicted Revenue : ${prediction:,.2f}"
    )

    if prediction >= df["Weekly_Sales"].mean():
        st.success("📈 High Revenue Expected")
    else:
        st.warning("📉 Low Revenue Expected")

        # ==========================================
    # Business Suggestions
    # ==========================================

    st.subheader("💡 Business Suggestions")

    average_revenue = df["Weekly_Sales"].mean()

    if prediction >= average_revenue * 1.20:
        st.success("🟢 Excellent Revenue Forecast")

        st.info("""
        ✅ Increase inventory

        ✅ Prepare additional staff

        ✅ Launch premium product promotions

        ✅ Focus on customer retention
        """)

    elif prediction >= average_revenue:

        st.success("🟡 Good Revenue Forecast")

        st.info("""
        ✅ Maintain current inventory

        ✅ Run seasonal offers

        ✅ Improve customer engagement

        ✅ Monitor weekly sales
        """)

    else:

        st.warning("🔴 Low Revenue Forecast")

        st.info("""
        ✅ Offer discounts

        ✅ Improve marketing

        ✅ Reduce unnecessary inventory

        ✅ Focus on high-demand products
        """)

    st.divider()

    # ==========================================
    # KPI Cards
    # ==========================================

    st.subheader("📊 Revenue Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💰 Predicted Revenue",
            f"${prediction:,.0f}"
        )

    with col2:
        st.metric(
            "📈 Average Revenue",
            f"${average_revenue:,.0f}"
        )

    with col3:
        status = "High" if prediction >= average_revenue else "Low"
        st.metric(
            "🎯 Forecast",
            status
        )

    st.divider()

    # ==========================================
    # Top 10 Stores by Revenue
    # ==========================================

    st.subheader("🏆 Top 10 Stores by Average Revenue")

    top10 = (
        df.groupby("Store")["Weekly_Sales"]
          .mean()
          .sort_values(ascending=False)
          .head(10)
    )

    fig, ax = plt.subplots(figsize=(5,3))

    bars = ax.barh(
        top10.index.astype(str),
        top10.values
    )

    ax.invert_yaxis()

    ax.set_xlabel("Average Weekly Revenue")
    ax.set_ylabel("Store")
    ax.set_title("Top 10 Performing Stores")

    for bar in bars:
        width = bar.get_width()

        ax.text(
            width,
            bar.get_y() + bar.get_height()/2,
            f"${width:,.0f}",
            va="center",
            fontsize=4
        )

    st.pyplot(fig)

    st.divider()

    # ==========================================
    # Revenue Performance Gauge
    # ==========================================

    st.subheader("🎯 Revenue Performance Meter")

    max_revenue = df["Weekly_Sales"].max()

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=prediction,

        number={
            "prefix":"$",
            "valueformat":",.0f"
        },

        title={
            "text":"Predicted Revenue"
        },

        gauge={

            "axis":{
                "range":[0, max_revenue*1.2]
            },

            "bar":{
                "color":"purple"
            },

            "steps":[

                {
                    "range":[0, average_revenue*0.8],
                    "color":"red"
                },

                {
                    "range":[average_revenue*0.8,
                             average_revenue*1.1],
                    "color":"yellow"
                },

                {
                    "range":[average_revenue*1.1,
                             max_revenue*1.2],
                    "color":"green"
                }

            ]

        }

    ))

    gauge.update_layout(height=380)

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

        # ==========================================
    # Monthly Revenue Analysis
    # ==========================================

    st.divider()
    st.subheader("📅 Monthly Revenue Analysis")

    monthly = (
        df.groupby("Month")["Weekly_Sales"]
          .mean()
          .reset_index()
    )

    month_names = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    monthly["Month_Name"] = monthly["Month"].apply(
        lambda x: month_names[x-1]
    )

    fig, ax = plt.subplots(figsize=(5,3))

    bars = ax.bar(
        monthly["Month_Name"],
        monthly["Weekly_Sales"]
    )

    ax.set_title("Average Revenue by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x()+bar.get_width()/2,
            height,
            f"{height:,.0f}",
            ha="center",
            fontsize=4
        )

    st.pyplot(fig)

    # ==========================================
    # Historical Revenue Trend
    # ==========================================

    st.divider()
    st.subheader("📈 Historical Revenue Trend")

    history = (
        df[df["Store"] == store]
        .sort_values("Date")
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(
        history["Date"],
        history["Weekly_Sales"],
        marker="o",
        linewidth=2,
        label="Historical Revenue"
    )

    ax.scatter(
        future_date,
        prediction,
        s=75,
        color="red",
        label="Prediction"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ==========================================
    # Revenue Distribution
    # ==========================================

    st.divider()
    st.subheader("📊 Revenue Distribution")

    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(
        df["Weekly_Sales"],
        bins=20
    )

    ax.set_title("Revenue Distribution")
    ax.set_xlabel("Weekly Revenue")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ==========================================
    # Download Prediction
    # ==========================================

    st.divider()
    st.subheader("📥 Download Prediction Report")

    report = pd.DataFrame({

        "Store":[store],

        "Prediction Date":[future_date],

        "Predicted Revenue":[prediction]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download CSV",

        csv,

        file_name="prediction_report.csv",

        mime="text/csv"

    )

    st.success("✅ Dashboard Generated Successfully")