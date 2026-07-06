# 📈 Business Revenue Forecasting System

A Machine Learning-based Business Revenue Forecasting System built using the Walmart Sales dataset. This project predicts future weekly revenue for a selected store using historical sales data and presents the results through an interactive Streamlit dashboard.

---

## 🚀 Features

- Predict future business revenue
- Store-wise revenue forecasting
- Future date prediction
- Automatic feature extraction
- Interactive Streamlit dashboard
- Revenue comparison charts
- Historical revenue visualization
- Business performance insights
- Revenue status (High / Low)
- Business suggestions based on prediction

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Business_Revenue_Forecasting/
│
├── app.py
├── train_model.py
├── feature_engineering.py
├── data_cleaning.py
├── requirements.txt
│
├── models/
│   └── revenue_model.pkl
│
├── outputs/
│   ├── featured_walmart.csv
│   └── feature_importance.csv
│
├── dataset/
│   └── walmart.csv
│
└── README.md
```

---

## 📊 Dataset

- Walmart Weekly Sales Dataset
- Historical data from **2010 to 2012**
- Contains multiple stores with weekly revenue and business-related features.

---

## 📌 Machine Learning Model

**Algorithm**

- Random Forest Regressor

**Features Used**

- Store
- Holiday Flag
- Temperature
- Fuel Price
- CPI
- Unemployment
- Year
- Month
- Week
- Quarter
- Day
- DayOfWeek
- IsWeekend
- Previous Week Sales
- Rolling Mean (4 Weeks)
- Rolling Standard Deviation
- Sales Difference
- Revenue Growth

**Target**

- Weekly Sales

---

## 📈 Model Performance

| Metric | Value |
|---------|--------|
| MAE | 8693.23 |
| RMSE | 18342.28 |
| R² Score | 0.9978 |

---

## 💻 Installation

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Dashboard Features

- Revenue Prediction
- Revenue Status
- Historical Revenue Trend
- Revenue Comparison Charts
- Business Suggestions
- Future Revenue Forecast

---

## 🎯 Future Improvements

- XGBoost Model
- LSTM Time Series Forecasting
- Prophet Forecasting
- PDF Report Generation
- Email Report
- Interactive Power BI Dashboard
- Cloud Deployment

---

## 👩‍💻 Author

**Mahalakshmi**

Aspiring Data Scientist | Machine Learning | Data Analytics | Streamlit | Python

---

## ⭐ If you found this project useful, consider giving it a star!