import dns.resolver
import os
import pandas as pd
import pymongo
import certifi
import pickle
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# --- 1. DNS PATCH ---
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']

# 2. Setup
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aqi_project_db"]
data_collection = db["pollution_data"] 
model_collection = db["model_registry"]

def train_and_save():
    print("🥊 Starting Daily Model Tournament...")
    
    # 3. Fetch Data
    data = list(data_collection.find().sort("timestamp", 1)) 
    if not data:
        print("❌ No data found!")
        return

    df = pd.DataFrame(data)
    
    # --- 🧠 FEATURE ENGINEERING ---
    df = df.sort_values("timestamp")
    df["pm2_5_lag1"] = df["pm2_5"].shift(1)
    df = df.dropna(subset=["pm2_5", "temp", "humidity", "wind_speed", "hour", "pm2_5_lag1"])
    
    # 4. Prepare Features
    features = ["temp", "humidity", "wind_speed", "hour", "pm2_5_lag1"]
    X = df[features]
    y = df["pm2_5"]
    
    # --- ✂️ TIME-SERIES SPLIT (Prevent Overfitting) ---
    # We take the first 80% for training and last 20% for testing
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 5. The Tournament
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_name = None
    best_model = None
    best_rmse = float("inf") 
    best_metrics = {}
    leaderboard = [] 

    print("-" * 75)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 75)

    for name, model in models.items():
        # Train on Training Set
        model.fit(X_train, y_train)
        
        # Predict on UNSEEN Test Set
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"{name:<20} | {mae:.4f}     | {rmse:.4f}     | {r2:.4f}")
        
        leaderboard.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
        
        # Winner is based on lowest RMSE on the TEST set
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model
            best_metrics = {"mae": mae, "rmse": rmse, "r2": r2}
            
    print("-" * 75)
    print(f"🏆 WINNER: {best_name}")

    # 6. Save Winner and Leaderboard
    model_data = {
        "model_name": best_name,
        "timestamp": datetime.datetime.now(),
        "metrics": best_metrics,
        "leaderboard": leaderboard,
        "model_binary": pickle.dumps(best_model)
    }
    
    model_collection.insert_one(model_data)
    print("✅ Champion Model and Leaderboard saved to Registry!")

if __name__ == "__main__":
    train_and_save()