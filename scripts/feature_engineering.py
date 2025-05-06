# feature_engineering.py

import pandas as pd
import numpy as np


# Set file paths
DATA_PATH = "your_data_path"

# Load cleaned training data
train = pd.read_csv(DATA_PATH + "train_clean.csv", parse_dates=["date"])
oil = pd.read_csv(DATA_PATH + "oil_clean.csv", parse_dates=["date"])
transactions = pd.read_csv(DATA_PATH + "transactions_clean.csv", parse_dates=["date"])
stores = pd.read_csv(DATA_PATH + "stores.csv")
holidays = pd.read_csv(DATA_PATH + "holidays_events.csv", parse_dates=["date"])


# Merge data
# Rename columns before merge to avoid conflict
stores.rename(columns={"type": "store_type"}, inplace=True)
holidays.rename(columns={"type": "holiday_type"}, inplace=True)


# Merge all datasets
df = train.merge(stores, on="store_nbr", how="left")
df = df.merge(transactions, on=["date", "store_nbr"], how="left")
df = df.merge(oil, on="date", how="left")
df = df.merge(holidays[["date", "holiday_type"]], on="date", how="left")


# ========== FEATURE ENGINEERING ==========

# 1. Time-based features
df["day"] = df["date"].dt.day
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 2. Lag features (by store + item)
df = df.sort_values(by=["store_nbr", "family", "date"])

# Lag of 7 days
df["lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)

# Rolling mean over past 7 days
df["rolling_mean_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(1).rolling(7).mean()
df["rolling_std_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(1).rolling(7).std()


# 3. Fill missing lags/rollings with 0 (or use forward fill)
df.fillna(0, inplace=True)

# 4. Encode categorical features
# One-hot encode categorical columns
df = pd.get_dummies(
    df, 
    columns=["family", "holiday_type", "store_type", "city", "state"], 
    drop_first=True
)


# 5. Final cleaning: drop unused columns
df.drop(["date", "id"], axis=1, errors="ignore", inplace=True)

# ========== SAVE FEATURES ==========


df.to_csv(DATA_PATH + "train_features.csv", index=False)
print("Feature engineering complete. Saved to train_features.csv")
