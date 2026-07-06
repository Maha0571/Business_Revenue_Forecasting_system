# ==========================================
# Business Revenue Forecasting System
# Model Training
# ==========================================

import pandas as pd
import pickle


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# Load Feature Engineered Dataset
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

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training Data :", X_train.shape)

print("Testing Data :", X_test.shape)

# ------------------------------------------
# Create Model
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ------------------------------------------
# Prediction
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluation Metrics
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print(f"Mean Absolute Error : {mae:.2f}")

print(f"Mean Squared Error : {mse:.2f}")

print(f"Root Mean Squared Error : {rmse:.2f}")

print(f"R2 Score : {r2:.4f}")

# ------------------------------------------
# Feature Importance
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(importance)

# ------------------------------------------
# Save Feature Importance
# ------------------------------------------

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nFeature Importance Saved!")

# ------------------------------------------
# Save Model
# ------------------------------------------

with open(
    "models/revenue_model.pkl",
    "wb"
) as file:

    pickle.dump(model, file)

print("\nModel Saved Successfully!")

print("Location : models/revenue_model.pkl")

# ------------------------------------------
# Completed
# ------------------------------------------

print("\n" + "=" * 60)

print("MODEL TRAINING COMPLETED SUCCESSFULLY")

print("=" * 60)
