# data_exploration.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set file paths
DATA_PATH = "your_data_path"

# 1. Load datasets
train = pd.read_csv(f"{DATA_PATH}train.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA_PATH}stores.csv")
oil = pd.read_csv(f"{DATA_PATH}oil.csv", parse_dates=["date"])
transactions = pd.read_csv(f"{DATA_PATH}transactions.csv", parse_dates=["date"])
holidays = pd.read_csv(f"{DATA_PATH}holidays_events.csv", parse_dates=["date"])


# 2. Basic info and structure
print("\n--- Train Dataset ---")
print(train.info())
print(train.describe())
print(train.head())


# 3. Check for missing values
print("\n--- Missing Values ---")
print(train.isnull().sum())


# 4. Merge auxiliary data for EDA
merged = train.merge(stores, on="store_nbr", how="left")
merged = merged.merge(transactions, on=["date", "store_nbr"], how="left")
merged = merged.merge(oil, on="date", how="left")
merged = merged.merge(holidays, on="date", how="left", suffixes=("", "_holiday"))


# 5. EDA - Plot total sales over time
plt.figure(figsize=(12, 6))
sales_by_date = train.groupby("date")["sales"].sum()
sales_by_date.plot()
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.close()


# 6. Check for duplicates
duplicates = train.duplicated().sum()
print(f"Duplicates in train: {duplicates}")


# 7. Clean any anomalies
# Remove negative or zero sales (invalid)
initial_rows = train.shape[0]
train = train[train["sales"] > 0]
print(f"Removed {initial_rows - train.shape[0]} rows with zero or negative sales.")


# 8. Fill missing values
# Fill missing oil prices using forward-fill
oil["dcoilwtico"] = oil["dcoilwtico"].fillna(method="ffill")

# Fill missing transactions with 0
transactions["transactions"] = transactions["transactions"].fillna(0)


# 9. Save cleaned datasets 
train.to_csv(DATA_PATH + "train_clean.csv", index=False)
oil.to_csv(DATA_PATH + "oil_clean.csv", index=False)
transactions.to_csv(DATA_PATH + "transactions_clean.csv", index=False)

print("Cleaned datasets saved.")
