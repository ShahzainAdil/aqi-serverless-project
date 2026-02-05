import os
import pickle
import datetime
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

# Import 3 different models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def get_database():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    return client.get_database("aqi_project_db")

def load_dataframe(db) -> pd.DataFrame:
    collection = db.get_collection("features")
    docs = list(collection.find({}))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def prepare_features(df: pd.DataFrame):
    df = df.copy()
    # Target: PM2.5 in the NEXT hour
    df["target_pm2_5"] = df["pm2_5"].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    
    # Feature Engineering: Add "Hour" column
    df["hour"] = df["timestamp"].dt.hour
    
    features = ["pm2_5", "pm10", "no2", "temp", "humidity", "wind_speed", "hour"]
    df = df[features + ["target_pm2_5"]]
    df = df.dropna()
    
    X = df[features]
    y = df["target_pm2_5"]
    return X, y

def train_and_evaluate(db, X, y):
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the 3 Contenders
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model_name = None
    best_mae = float("inf")
    best_model_obj = None
    
    print(f"🥊 Starting Model Tournament with {len(X)} records...")
    print("-" * 40)

    results = []

    # Train and Evaluate each
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"   🏃 {name}: MAE = ±{mae:.4f}")
        
        results.append({
            "model_name": name,
            "mae": mae,
            "trained_at": datetime.datetime.utcnow()
        })
        
        # Check if this is the new winner
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model_obj = model

    print("-" * 40)
    print(f"🏆 WINNER: {best_model_name} (Error: ±{best_mae:.4f})")
    
    # 1. Save the Winner to File (for the App to use)
    with open("aqi_model.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
        
    # 2. Log the Tournament Results to MongoDB (Model Registry)
    registry_collection = db.get_collection("model_registry")
    
    # Create a registry document
    registry_doc = {
        "experiment_date": datetime.datetime.utcnow(),
        "winner": best_model_name,
        "winner_mae": best_mae,
        "candidates": results
    }
    registry_collection.insert_one(registry_doc)
    print("✅ Model Registry updated in MongoDB.")

def main():
    db = get_database()
    df = load_dataframe(db)
    
    if df.empty:
        print("No data found.")
        return
        
    X, y = prepare_features(df)
    train_and_evaluate(db, X, y)

if __name__ == "__main__":
    main()