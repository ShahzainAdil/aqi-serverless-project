import os
import datetime
import requests
import pymongo
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
MONGO_URI = os.getenv("MONGO_URI")
# Karachi Coordinates
LAT = 24.8607
LON = 67.0011

def get_collection():
    if not MONGO_URI:
        raise ValueError("MONGO_URI is missing from .env file")
    client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["aqi_project_db"]
    return db["features"]

def fetch_open_meteo_data():
    """Fetch current weather and air quality from Open-Meteo (No Key Needed)."""
    print("🌍 Fetching live data from Open-Meteo...")
    
    # 1. Weather API
    weather_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    w_resp = requests.get(weather_url, params=w_params)
    w_resp.raise_for_status()
    w_data = w_resp.json()['current']

    # 2. Air Quality API
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "pm2_5,pm10,nitrogen_dioxide",
        "timezone": "auto"
    }
    aq_resp = requests.get(aq_url, params=aq_params)
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()['current']

    # 3. Create Timestamp (Rounded to nearest hour)
    current_utc = datetime.datetime.now(datetime.timezone.utc)
    rounded_time = current_utc.replace(minute=0, second=0, microsecond=0)

    # 4. Build the Data Dictionary
    doc = {
        "timestamp": rounded_time,
        "pm2_5": aq_data["pm2_5"],
        "pm10": aq_data["pm10"],
        "no2": aq_data["nitrogen_dioxide"],
        "temp": w_data["temperature_2m"],
        "humidity": w_data["relative_humidity_2m"],
        "wind_speed": w_data["wind_speed_10m"],
        "hour": rounded_time.hour  # Helper for ML
    }
    
    return doc

def save_to_mongo(data):
    collection = get_collection()
    
    # "Upsert": If data for this hour exists, update it. If not, insert new.
    # This prevents duplicate data for the same hour.
    result = collection.replace_one(
        {"timestamp": data["timestamp"]}, 
        data, 
        upsert=True
    )
    
    if result.matched_count > 0:
        print(f"🔄 Updated record for: {data['timestamp']}")
    else:
        print(f"✅ Inserted new record for: {data['timestamp']}")

if __name__ == "__main__":
    try:
        data = fetch_open_meteo_data()
        save_to_mongo(data)
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")