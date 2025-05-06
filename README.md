# 📦 Demand Forecasting System

A Machine Learning application that predicts **weekly product sales** based on **store type**, **product family**, and **promotion status** using **XGBoost** and **time series features**. The goal is to help businesses forecast product demand a week ahead to improve inventory planning and reduce stockouts or overstocking.

Built with a full pipeline: from raw data to a real-time interactive app using **Streamlit**.

---

## ❓ What This Project Does

- Predicts weekly sales for a selected **product type** and **store type**
- Uses real-world historical data (Ecuador grocery chain from Kaggle)
- Includes time-based feature engineering for better forecasting
- Outputs:
  - Total sales prediction for next week
  - Weekly trend chart
  - Prediction table

---

## 🎯 Why This Project

- Simulates a real-world business use case
- Demonstrates skills in:
  - Machine learning (XGBoost)
  - Time series forecasting
  - Feature engineering
  - Streamlit UI design
  - Clean code and pipeline structure


---

## ⚙️ Tech Stack

| Category        | Tools Used             |
|----------------|------------------------|
| ML Model       | XGBoost (Regressor)    |
| Time Series    | Feature Engineering    |
| Web App        | Streamlit              |
| Language       | Python                 |
| Data Handling  | Pandas, NumPy          |
| Visualization  | Matplotlib, Streamlit Charts |
| Model Evaluation | RMSLE                 |

---

## 📊 Dataset Used

- Source: [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- Description: Historical sales data for a grocery chain in Ecuador
- Files: `train.csv`, `stores.csv`, `items.csv`, `oil.csv`, `holidays_events.csv`
- Note: Due to size and license, **not included in repo** – must be downloaded manually.

---
