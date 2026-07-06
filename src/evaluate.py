# ==========================================
# Business Revenue Forecasting System
# Model Evaluation
# ==========================================

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("outputs/featured_walmart.csv")

# ------------------------------------------
# Select Features
# ------------------------------------------

X = df[
    [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Year",
        "Month",
        "Week",
        "Quarter",
        "Day",
        "DayOfWeek",
        "IsWeekend",
        "Previous_Week_Sales",
        "Rolling_Mean_4",
        "Rolling_STD_4",
        "Sales_Difference",
        "Revenue_Growth"
    ]
]

# ------------------------------------------
# Target Variable
# ------------------------------------------

y = df["Weekly_Sales"]

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------
# Load Trained Model
# ------------------------------------------

with open("models/revenue_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------------------------------
# Predict
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluation Metrics
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

comparison = pd.DataFrame({
    "Actual Revenue": y_test.values,
    "Predicted Revenue": y_pred
})

print("\nActual vs Predicted")

print(comparison.head(15))

# ------------------------------------------
# Save Prediction Result
# ------------------------------------------

comparison.to_csv(
    "outputs/revenue_prediction.csv",
    index=False
)

print("\nPrediction File Saved Successfully!")

# ------------------------------------------
# Actual vs Predicted Scatter Plot
# ------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.title("Actual vs Predicted Revenue")

plt.xlabel("Actual Revenue")

plt.ylabel("Predicted Revenue")

plt.grid(True)

plt.show()

# ------------------------------------------
# Actual vs Predicted Line Graph
# ------------------------------------------

comparison = comparison.reset_index(drop=True)

plt.figure(figsize=(14,6))

plt.plot(
    comparison["Actual Revenue"][:50],
    label="Actual Revenue"
)

plt.plot(
    comparison["Predicted Revenue"][:50],
    label="Predicted Revenue"
)

plt.title("Actual vs Predicted Revenue")

plt.xlabel("Records")

plt.ylabel("Revenue")

plt.legend()

plt.grid(True)

plt.show()

# ------------------------------------------
# Prediction Error
# ------------------------------------------

comparison["Error"] = (
    comparison["Actual Revenue"]
    -
    comparison["Predicted Revenue"]
)

print("\nPrediction Error")

print(comparison.head())

# ------------------------------------------
# Feature Importance
# ------------------------------------------

importance = pd.read_csv(
    "outputs/feature_importance.csv"
)

print("\nFeature Importance")

print(importance)

# ------------------------------------------
# Feature Importance Graph
# ------------------------------------------

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.ylabel("Features")

plt.show()

# ------------------------------------------
# Completed
# ------------------------------------------

print("\n" + "=" * 60)

print("MODEL EVALUATION COMPLETED SUCCESSFULLY")

print("=" * 60)