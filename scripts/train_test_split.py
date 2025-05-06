# train_test_split.py

import pandas as pd


# Set file paths
DATA_PATH = "your_data_path"

# Load feature-engineered dataset
df = pd.read_csv(DATA_PATH + "train_features.csv")

# Target column
TARGET = "sales"

# Add date back for split reference (optional — only if you saved it earlier)
# If date was dropped before saving, re-load from original clean file and align
try:
    clean = pd.read_csv(DATA_PATH + "train_clean.csv", parse_dates=["date"])
    df["date"] = clean["date"]
except Exception as e:
    print("Warning: could not restore date column. Make sure 'date' is in the features file if needed.")
    raise e

# Split based on date
cutoff_date = "your_split_date"
train_df = df[df["date"] < cutoff_date].copy()
valid_df = df[df["date"] >= cutoff_date].copy()

# Drop the date column before training
train_df.drop(columns=["date"], inplace=True)
valid_df.drop(columns=["date"], inplace=True)

# Split into features (X) and target (y)
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_valid = valid_df.drop(columns=[TARGET])
y_valid = valid_df[TARGET]


# Save splits
X_train.to_csv(DATA_PATH + "X_train.csv", index=False)
y_train.to_csv(DATA_PATH + "y_train.csv", index=False)
X_valid.to_csv(DATA_PATH + "X_valid.csv", index=False)
y_valid.to_csv(DATA_PATH + "y_valid.csv", index=False)

print("Time-based train/test split completed.")
print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
