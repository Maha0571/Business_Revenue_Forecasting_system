# ==========================================
# Business Revenue Forecasting System
# Future Revenue Forecasting
# ==========================================

import pickle
import pandas as pd

# ------------------------------------------
# Load Trained Model
# ------------------------------------------

with open("models/revenue_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------------------
# Load Feature Engineered Dataset
# ------------------------------------------

df = pd.read_csv("outputs/featured_walmart.csv")

df["Date"] = pd.to_datetime(df["Date"])

# ------------------------------------------
# User Input
# ------------------------------------------

print("=" * 60)
print("BUSINESS REVENUE FORECASTING SYSTEM")
print("=" * 60)

store = int(input("Enter Store Number : "))
future_date = input("Enter Future Prediction Date (YYYY-MM-DD) : ")

future_date = pd.to_datetime(future_date)

# ------------------------------------------
# Validate Store
# ------------------------------------------

store_data = df[df["Store"] == store].sort_values("Date")

if store_data.empty:
    print("\nInvalid Store Number!")
    exit()

# ------------------------------------------
# Validate Prediction Date
# ------------------------------------------

dataset_start = df["Date"].min()
dataset_end = df["Date"].max()

# Date before dataset
if future_date < dataset_start:

    print("\nPrediction not possible!")
    print(f"Dataset starts from : {dataset_start.date()}")
    exit()

# ------------------------------------------
# Get Historical / Future Base Record
# ------------------------------------------

# Historical Prediction (2010–2012)
if future_date <= dataset_end:

    latest = store_data[store_data["Date"] <= future_date]

    if latest.empty:
        print("\nNo historical data available.")
        exit()

    latest = latest.iloc[-1]

# Future Forecast (>2012)
else:

    latest = store_data.iloc[-1]

latest_date = latest["Date"]

# ------------------------------------------
# Generate Date Features
# ------------------------------------------

year = future_date.year
month = future_date.month
week = int(future_date.isocalendar().week)
quarter = future_date.quarter
day = future_date.day
day_of_week = future_date.dayofweek
is_weekend = 1 if day_of_week >= 5 else 0

# ------------------------------------------
# Holiday Estimation
# ------------------------------------------

holiday = 1 if month in [11, 12] else 0

# ------------------------------------------
# Automatically Fetch Remaining Features
# ------------------------------------------

temperature = latest["Temperature"]
fuel_price = latest["Fuel_Price"]
cpi = latest["CPI"]
unemployment = latest["Unemployment"]

previous_sales = latest["Weekly_Sales"]

rolling_mean = latest["Rolling_Mean_4"]
rolling_std = latest["Rolling_STD_4"]

sales_difference = latest["Sales_Difference"]
revenue_growth = latest["Revenue_Growth"]

# ------------------------------------------
# Create Input Data
# ------------------------------------------

input_data = pd.DataFrame({

    "Store": [store],

    "Holiday_Flag": [holiday],

    "Temperature": [temperature],

    "Fuel_Price": [fuel_price],

    "CPI": [cpi],

    "Unemployment": [unemployment],

    "Year": [year],

    "Month": [month],

    "Week": [week],

    "Quarter": [quarter],

    "Day": [day],

    "DayOfWeek": [day_of_week],

    "IsWeekend": [is_weekend],

    "Previous_Week_Sales": [previous_sales],

    "Rolling_Mean_4": [rolling_mean],

    "Rolling_STD_4": [rolling_std],

    "Sales_Difference": [sales_difference],

    "Revenue_Growth": [revenue_growth]

})

# ------------------------------------------
# Predict Revenue
# ------------------------------------------

prediction = model.predict(input_data)[0]

# ------------------------------------------
# Revenue Status
# ------------------------------------------

average_revenue = df["Weekly_Sales"].mean()

if prediction >= average_revenue:

    status = "High Revenue"

    suggestions = [

        "Maintain the current pricing strategy.",

        "Increase inventory for high-demand products.",

        "Continue successful marketing campaigns.",

        "Focus on customer retention programs.",

        "Expand business to nearby locations."

    ]

else:

    status = "Low Revenue"

    suggestions = [

        "Launch promotional offers and discounts.",

        "Improve inventory management.",

        "Optimize product pricing strategy.",

        "Increase digital marketing campaigns.",

        "Analyze customer buying patterns.",

        "Improve customer service experience."

    ]

# ------------------------------------------
# Display Result
# ------------------------------------------

print("\n" + "=" * 60)
print("BUSINESS REVENUE FORECAST RESULT")
print("=" * 60)

print(f"Store Number                : {store}")
print(f"Forecast Date               : {future_date.date()}")
print(f"Latest Historical Date Used : {latest_date.date()}")

print("-" * 60)

print("Automatically Retrieved Features")

print(f"Holiday Flag        : {holiday}")
print(f"Temperature         : {temperature:.2f}")
print(f"Fuel Price          : {fuel_price:.2f}")
print(f"CPI                 : {cpi:.2f}")
print(f"Unemployment        : {unemployment:.2f}")
print(f"Previous Sales      : {previous_sales:.2f}")
print(f"Rolling Mean (4)    : {rolling_mean:.2f}")
print(f"Rolling Std (4)     : {rolling_std:.2f}")
print(f"Sales Difference    : {sales_difference:.2f}")
print(f"Revenue Growth (%)  : {revenue_growth:.2f}")

print("-" * 60)

print(f"Predicted Revenue   : ${prediction:,.2f}")
print(f"Revenue Status      : {status}")

print("-" * 60)

print("Business Suggestions")

for i, suggestion in enumerate(suggestions, start=1):
    print(f"{i}. {suggestion}")

print("=" * 60)
print("Future Revenue Forecast Completed Successfully!")
print("=" * 60)