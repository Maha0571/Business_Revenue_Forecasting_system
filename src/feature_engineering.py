# ==========================================
# Business Revenue Forecasting System
# Feature Engineering
# ==========================================

import pandas as pd

# ------------------------------------------
# Load Cleaned Dataset
# ------------------------------------------

df = pd.read_csv("outputs/cleaned_walmart.csv")

# ------------------------------------------
# Convert Date Column
# ------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

# ------------------------------------------
# Sort Dataset
# ------------------------------------------

df = df.sort_values(
    by=["Store", "Date"]
)

# ------------------------------------------
# Create Date Features
# ------------------------------------------

df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

df["Quarter"] = df["Date"].dt.quarter

df["Day"] = df["Date"].dt.day

df["DayOfWeek"] = df["Date"].dt.dayofweek

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

# ------------------------------------------
# Previous Week Sales (Lag Feature)
# ------------------------------------------

df["Previous_Week_Sales"] = (
    df.groupby("Store")["Weekly_Sales"]
    .shift(1)
)

# Fill Missing Values

df["Previous_Week_Sales"] = df[
    "Previous_Week_Sales"
].fillna(
    df["Weekly_Sales"].median()
)

# ------------------------------------------
# Rolling Mean (Last 4 Weeks)
# ------------------------------------------

df["Rolling_Mean_4"] = (
    df.groupby("Store")["Weekly_Sales"]
    .transform(
        lambda x: x.rolling(4).mean()
    )
)

df["Rolling_Mean_4"] = df[
    "Rolling_Mean_4"
].fillna(
    df["Weekly_Sales"].median()
)

# ------------------------------------------
# Rolling Standard Deviation
# ------------------------------------------

df["Rolling_STD_4"] = (
    df.groupby("Store")["Weekly_Sales"]
    .transform(
        lambda x: x.rolling(4).std()
    )
)

df["Rolling_STD_4"] = df[
    "Rolling_STD_4"
].fillna(0)

# ------------------------------------------
# Sales Difference
# ------------------------------------------

df["Sales_Difference"] = (
    df["Weekly_Sales"]
    -
    df["Previous_Week_Sales"]
)

# ------------------------------------------
# Revenue Growth (%)
# ------------------------------------------

df["Revenue_Growth"] = (
    (
        df["Weekly_Sales"]
        -
        df["Previous_Week_Sales"]
    )
    /
    df["Previous_Week_Sales"]
) * 100

df["Revenue_Growth"] = df[
    "Revenue_Growth"
].fillna(0)

# ------------------------------------------
# Display Dataset
# ------------------------------------------

print("=" * 60)

print("FEATURE ENGINEERING")

print("=" * 60)

print(df.head())

# ------------------------------------------
# Dataset Shape
# ------------------------------------------

print("\nDataset Shape")

print(df.shape)

# ------------------------------------------
# New Columns
# ------------------------------------------

print("\nColumns")

print(df.columns.tolist())

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\nMissing Values")

print(df.isnull().sum())

# ------------------------------------------
# Save Dataset
# ------------------------------------------

df.to_csv(
    "outputs/featured_walmart.csv",
    index=False
)

print("\nFeature Engineered Dataset Saved Successfully!")

print("Location : outputs/featured_walmart.csv")

# ------------------------------------------
# Completed
# ------------------------------------------

print("\n" + "=" * 60)

print("FEATURE ENGINEERING COMPLETED")

print("=" * 60)