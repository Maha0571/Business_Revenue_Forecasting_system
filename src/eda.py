# ==========================================
# Business Revenue Forecasting System
# Exploratory Data Analysis (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------
# Load Cleaned Dataset
# ------------------------------------------

df = pd.read_csv("outputs/cleaned_walmart.csv")

# Convert Date Column

df["Date"] = pd.to_datetime(df["Date"])

# ------------------------------------------
# Dataset Overview
# ------------------------------------------

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst Five Rows")

print(df.head())

print("\nShape")

print(df.shape)

print("\nColumns")

print(df.columns.tolist())

# ------------------------------------------
# Statistical Summary
# ------------------------------------------

print("\nStatistical Summary")

print(df.describe())

# ------------------------------------------
# Correlation Matrix
# ------------------------------------------

print("\nCorrelation Matrix")

print(df.corr(numeric_only=True))

# ------------------------------------------
# Revenue Summary
# ------------------------------------------

print("\nRevenue Summary")

print("Maximum Revenue :", df["Weekly_Sales"].max())

print("Minimum Revenue :", df["Weekly_Sales"].min())

print("Average Revenue :", round(df["Weekly_Sales"].mean(),2))

print("Median Revenue :", round(df["Weekly_Sales"].median(),2))

# ------------------------------------------
# Monthly Revenue
# ------------------------------------------

df["Month"] = df["Date"].dt.month

monthly_sales = df.groupby("Month")["Weekly_Sales"].mean()

print("\nMonthly Revenue")

print(monthly_sales)

plt.figure(figsize=(10,5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Average Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Store-wise Revenue
# ------------------------------------------

store_sales = df.groupby("Store")["Weekly_Sales"].sum()

print("\nStore-wise Revenue")

print(store_sales)

plt.figure(figsize=(15,6))

plt.bar(
    store_sales.index,
    store_sales.values
)

plt.title("Revenue by Store")

plt.xlabel("Store")

plt.ylabel("Revenue")

plt.xticks(rotation=90)

plt.show()

# ------------------------------------------
# Holiday Revenue
# ------------------------------------------

holiday_sales = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()

print("\nHoliday Revenue")

print(holiday_sales)

plt.figure(figsize=(6,5))

plt.bar(
    holiday_sales.index.astype(str),
    holiday_sales.values
)

plt.title("Holiday vs Non-Holiday Revenue")

plt.xlabel("Holiday Flag")

plt.ylabel("Average Revenue")

plt.show()

# ------------------------------------------
# Temperature vs Revenue
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["Temperature"],
    df["Weekly_Sales"]
)

plt.title("Temperature vs Revenue")

plt.xlabel("Temperature")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Fuel Price vs Revenue
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["Fuel_Price"],
    df["Weekly_Sales"]
)

plt.title("Fuel Price vs Revenue")

plt.xlabel("Fuel Price")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# CPI vs Revenue
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["CPI"],
    df["Weekly_Sales"]
)

plt.title("CPI vs Revenue")

plt.xlabel("CPI")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Unemployment vs Revenue
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["Unemployment"],
    df["Weekly_Sales"]
)

plt.title("Unemployment vs Revenue")

plt.xlabel("Unemployment")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Revenue Trend Over Time
# ------------------------------------------

trend = df.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(15,6))

plt.plot(
    trend.index,
    trend.values
)

plt.title("Revenue Trend Over Time")

plt.xlabel("Date")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Top 10 Stores
# ------------------------------------------

top10 = store_sales.sort_values(
    ascending=False
).head(10)

print("\nTop 10 Stores")

print(top10)

plt.figure(figsize=(10,5))

plt.bar(
    top10.index.astype(str),
    top10.values
)

plt.title("Top 10 Stores by Revenue")

plt.xlabel("Store")

plt.ylabel("Revenue")

plt.show()

# ------------------------------------------
# Bottom 10 Stores
# ------------------------------------------

bottom10 = store_sales.sort_values().head(10)

print("\nBottom 10 Stores")

print(bottom10)

plt.figure(figsize=(10,5))

plt.bar(
    bottom10.index.astype(str),
    bottom10.values
)

plt.title("Bottom 10 Stores by Revenue")

plt.xlabel("Store")

plt.ylabel("Revenue")

plt.show()

# ------------------------------------------
# Average Revenue by Year
# ------------------------------------------

df["Year"] = df["Date"].dt.year

year_sales = df.groupby("Year")["Weekly_Sales"].mean()

print("\nYear-wise Revenue")

print(year_sales)

plt.figure(figsize=(6,5))

plt.plot(
    year_sales.index,
    year_sales.values,
    marker="o"
)

plt.title("Average Revenue by Year")

plt.xlabel("Year")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# EDA Completed
# ------------------------------------------

print("\n" + "=" * 60)

print("EDA COMPLETED SUCCESSFULLY")

print("=" * 60)