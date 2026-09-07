"""
Stock Price Predictor - L23_2526
Predicts next-day closing price using RandomForestRegressor
with moving-average based features.

Expects data/dataset.csv with columns: Date, Open, High, Low, Close, Volume
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = "data/dataset.csv"
data = pd.read_csv(DATA_PATH)
print("Dataset loaded:", data.shape)

# -----------------------------
# 2. Feature engineering
# -----------------------------
data["MA5"] = (data["Close"].rolling(window=5).mean() - data["Close"].min()) / (data["Close"].max() - data["Close"].min())
data["MA10"] = data["Close"].rolling(window=10).mean()

# Target: next day's closing price
data["Target"] = data["Close"].shift(-1)

# Drop rows with NaNs created by rolling windows / shift
data = data.dropna()

FEATURES = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10"]
X = data[FEATURES]
y = data["Target"]

# -----------------------------
# 3. Train/test split (no shuffle — time series)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 4. Train model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.2f}")

# -----------------------------
# 6. Save model artifact
# -----------------------------
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/model_L23_2526.pkl"
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")