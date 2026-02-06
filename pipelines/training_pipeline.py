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
feature_collection = db["features"]
model_collection = db["model_registry"]

def train_and_save():
    print("🥊 Starting Daily Model Tournament...")
    
    # 3. Fetch Data
    data = list(feature_collection.find())
    if not data:
        print("❌ No data found!")
        return

    df = pd.DataFrame(data)
    
    # Clean Data
    required_cols = ["pm2_5", "temp", "humidity", "wind_speed", "hour"]
    df = df.dropna(subset=required_cols)
    
    # 4. Prepare Features
    X = df[["temp", "humidity", "wind_speed", "hour"]]
    y = df["pm2_5"]
    
    # 5. The Tournament
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_name = None
    best_model = None
    best_r2 = -float("inf") # We want to maximize R2
    best_metrics = {}
    
    print("-" * 75)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 75)

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        
        # --- NEW METRICS ---
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        
        print(f"{name:<20} | {mae:.4f}     | {rmse:.4f}     | {r2:.4f}")
        
        # We choose the winner based on R2 (Accuracy)
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
            best_metrics = {"mae": mae, "rmse": rmse, "r2": r2}
            
    print("-" * 75)
    print(f"🏆 WINNER: {best_name} (Accuracy: {best_metrics['r2']*100:.2f}%)")

    # 6. Save Winner
    model_data = {
        "model_name": best_name,
        "timestamp": datetime.datetime.now(),
        "metrics": best_metrics,
        "model_binary": pickle.dumps(best_model)
    }
    
    model_collection.delete_many({}) 
    model_collection.insert_one(model_data)
    print("✅ Champion Model saved to Registry!")

if __name__ == "__main__":
    train_and_save()