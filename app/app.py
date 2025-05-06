# app.py

import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import datetime

# Set file paths
DATA_PATH = "your_data_path"


# Load model and feature metadata
model = xgb.Booster()
model.load_model(DATA_PATH + "models/xgb_model.json")
feature_names = joblib.load(DATA_PATH + "models/feature_names.pkl")


# Load original cleaned dataset (before encoding)
df = pd.read_csv(DATA_PATH + "train_clean.csv")


# Sidebar Inputs
st.sidebar.header("📊 Input Parameters")
family = st.sidebar.selectbox("Select Product Family", sorted(df["family"].unique()))
store_nbr = st.sidebar.selectbox("Select Store Number", sorted(df["store_nbr"].unique()))
onpromotion = st.sidebar.radio("Is the Product on Promotion?", [0, 1])


# Determine next week's date range
today = datetime.date.today()
next_monday = today + datetime.timedelta(days=(7 - today.weekday()))
next_sunday = next_monday + datetime.timedelta(days=6)


# Build input row for prediction
row = df.iloc[-1:].copy()
row["family"] = family
row["store_nbr"] = store_nbr
row["onpromotion"] = onpromotion
row["date"] = pd.to_datetime(next_monday)


# Add date-based features
row["day_of_week"] = next_monday.weekday()
row["month"] = next_monday.month
row["year"] = next_monday.year
row["is_weekend"] = 1 if next_monday.weekday() >= 5 else 0


# One-hot encode & align with training features
row_encoded = pd.get_dummies(row)
row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0)


# Make prediction
dmatrix = xgb.DMatrix(row_encoded)
predicted_sales = model.predict(dmatrix)[0]


# Format output
week_label = f"{next_monday.strftime('%d %b')} → {next_sunday.strftime('%d %b')}"
forecast_df = pd.DataFrame([{
    "Week": week_label,
    "Predicted Sales": round(predicted_sales, 2)
}])


# Display
st.title("🛍️ Next Week Demand Forecast")
st.subheader(f"📦 Product: `{family}` | Store #: `{store_nbr}` | Promo: `{onpromotion}`")

st.write("### 🔢 Forecast Table")
st.table(forecast_df)

st.metric(label="📈 Total Forecasted Sales", value=f"{predicted_sales:.2f} units")

