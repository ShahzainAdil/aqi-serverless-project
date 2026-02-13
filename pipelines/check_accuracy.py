import pandas as pd
import pymongo
import certifi
import os
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dotenv import load_dotenv
import datetime

# 1. Setup
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aqi_project_db"]

def check_live_performance():
    print("🕵️ Checking model accuracy on REAL PAST data...")

    # 2. Load Model
    model_doc = db["model_registry"].find_one(sort=[("timestamp", -1)])
    if not model_doc:
        print("❌ No model found!")
        return
    model = pickle.loads(model_doc["model_binary"])

    # 3. Fetch Data
    # Get ALL data, then we filter in Python to be safe
    data = list(db["features"].find())
    df = pd.DataFrame(data)
    
    # 4. Strict Filtering: "The Reality Check"
    # We only want rows where the time is BEFORE right now
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    
    # Ensure database timestamps are timezone-aware (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None) # Strip TZ for comparison
    now_utc = now_utc.replace(tzinfo=None) # Strip TZ for comparison

    # FILTER: Keep only Past Data
    past_df = df[df["timestamp"] < now_utc].copy()
    
    # Sort by newest first and take last 24 hours
    past_df = past_df.sort_values("timestamp", ascending=False).head(24)
    
    if past_df.empty:
        print("❌ No past data found! (Check your database timestamps)")
        return

    print(f"✅ Found {len(past_df)} valid historical records.")

    # 5. Prepare & Predict
    feature_cols = ["temp", "humidity", "wind_speed", "hour"]
    past_df = past_df.dropna(subset=feature_cols + ["pm2_5"])
    
    X_new = past_df[feature_cols]
    y_actual = past_df["pm2_5"]
    
    predictions = model.predict(X_new)
    
    # 6. Calculate Error
    mae = mean_absolute_error(y_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    
    # 7. Report
    print("\n" + "="*50)
    print(f"📊 TRUE PERFORMANCE REPORT (Last 24 REAL Hours)")
    print("="*50)
    print(f"True Error (MAE):  ±{mae:.2f}")
    print(f"True Error (RMSE): ±{rmse:.2f}")
    print("-" * 50)
    
    results = pd.DataFrame({
        "Time": past_df["timestamp"],
        "Actual": y_actual,
        "Predicted": predictions,
        "Diff": abs(y_actual - predictions)
    })
    print(results.head(10).to_string(index=False))

if __name__ == "__main__":
    check_live_performance()