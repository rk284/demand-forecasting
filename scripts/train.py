# train.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
import joblib



# Set file paths
DATA_PATH = "your_data_path"


# Load data
X_train = pd.read_csv(DATA_PATH + "X_train.csv")
y_train = pd.read_csv(DATA_PATH + "y_train.csv").squeeze()
X_valid = pd.read_csv(DATA_PATH + "X_valid.csv")
y_valid = pd.read_csv(DATA_PATH + "y_valid.csv").squeeze()

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# Define training parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "seed": 42
}

# Train model with early stopping
evallist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Predict and evaluate
y_pred = model.predict(dvalid)

def rmsle(y_true, y_pred):
    y_pred = [max(0, p) for p in y_pred]
    return mean_squared_log_error(y_true, y_pred) ** 0.5

score = rmsle(y_valid, y_pred)
print(f"RMSLE on validation set: {score:.4f}")


# Save model and features
model.save_model(DATA_PATH + "/models/xgb_model.json")
joblib.dump(X_train.columns.tolist(), f"{DATA_PATH}/models/feature_names.pkl")
print("Model and feature names saved in models/")
