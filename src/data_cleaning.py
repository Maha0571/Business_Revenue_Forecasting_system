# ==========================================
# Business Revenue Forecasting System
# Data Cleaning
# ==========================================

import pandas as pd
import numpy as np

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("data/walmart-sales-dataset-of-45stores.csv")

# ------------------------------------------
# Display Dataset
# ------------------------------------------

print("=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())

# ------------------------------------------
# Dataset Information
# ------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

# ------------------------------------------
# Dataset Shape
# ------------------------------------------

print("\nDataset Shape")

print(df.shape)

# ------------------------------------------
# Column Names
# ------------------------------------------

print("\nColumns")

print(df.columns.tolist())

# ------------------------------------------
# Check Missing Values
# ------------------------------------------

print("\nMissing Values")

print(df.isnull().sum())

# ------------------------------------------
# Check Duplicate Rows
# ------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate Rows Removed Successfully!")
else:
    print("No Duplicate Rows Found.")

# ------------------------------------------
# Convert Date Column
# ------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

print("\nDate Column Converted Successfully!")

# ------------------------------------------
# Check Data Types
# ------------------------------------------

print("\nData Types")

print(df.dtypes)

# ------------------------------------------
# Basic Statistics
# ------------------------------------------

print("\nStatistical Summary")

print(df.describe())

# ------------------------------------------
# Unique Stores
# ------------------------------------------

print("\nTotal Stores")

print(df["Store"].nunique())

# ------------------------------------------
# Holiday Flag Count
# ------------------------------------------

print("\nHoliday Flag Distribution")

print(df["Holiday_Flag"].value_counts())

# ------------------------------------------
# Weekly Sales Summary
# ------------------------------------------

print("\nWeekly Sales Summary")

print("Maximum Sales :", df["Weekly_Sales"].max())

print("Minimum Sales :", df["Weekly_Sales"].min())

print("Average Sales :", round(df["Weekly_Sales"].mean(), 2))

print("Median Sales :", round(df["Weekly_Sales"].median(), 2))

# ------------------------------------------
# Temperature Summary
# ------------------------------------------

print("\nTemperature Summary")

print("Maximum :", df["Temperature"].max())

print("Minimum :", df["Temperature"].min())

# ------------------------------------------
# Fuel Price Summary
# ------------------------------------------

print("\nFuel Price Summary")

print("Maximum :", df["Fuel_Price"].max())

print("Minimum :", df["Fuel_Price"].min())

# ------------------------------------------
# CPI Summary
# ------------------------------------------

print("\nCPI Summary")

print("Maximum :", df["CPI"].max())

print("Minimum :", df["CPI"].min())

# ------------------------------------------
# Unemployment Summary
# ------------------------------------------

print("\nUnemployment Summary")

print("Maximum :", df["Unemployment"].max())

print("Minimum :", df["Unemployment"].min())

# ------------------------------------------
# Check Negative Values
# ------------------------------------------

print("\nNegative Value Check")

numeric_columns = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment"
]

for column in numeric_columns:

    negative_values = (df[column] < 0).sum()

    print(f"{column} : {negative_values}")

# ------------------------------------------
# Sort Dataset by Date
# ------------------------------------------

df = df.sort_values(
    by=["Store", "Date"]
)

df = df.reset_index(drop=True)

print("\nDataset Sorted Successfully!")

# ------------------------------------------
# Save Cleaned Dataset
# ------------------------------------------

df.to_csv(
    "outputs/cleaned_walmart.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")

print("\nLocation : outputs/cleaned_walmart.csv")

# ------------------------------------------
# Completed
# ------------------------------------------

print("\n" + "=" * 60)

print("DATA CLEANING COMPLETED SUCCESSFULLY")

print("=" * 60)