import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import pymongo
import certifi
import os
import datetime
from dotenv import load_dotenv

# 1. Setup Database Connection
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aqi_project_db"]
collection = db["features"]

# Setup Open-Meteo API Client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_latest_hour():
    print("📡 Connecting to Open-Meteo for latest data...")
    
    # 2. Coordinates for Karachi
    LAT = 24.8607
    LON = 67.0011
    
    # We ask for "past_days=1" to ensure we get the most recent completed hour
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aq = {
        "latitude": LAT, "longitude": LON,
        "hourly": ["pm2_5", "pm10", "nitrogen_dioxide"],
        "past_days": 1
    }
    
    url_w = "https://api.open-meteo.com/v1/forecast"
    params_w = {
        "latitude": LAT, "longitude": LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "past_days": 1
    }

    try:
        # Fetch Air Quality
        aq_resp = openmeteo.weather_api(url_aq, params=params_aq)[0]
        hourly_aq = aq_resp.Hourly()
        
        # Fetch Weather
        w_resp = openmeteo.weather_api(url_w, params=params_w)[0]
        hourly_w = w_resp.Hourly()

        # 3. Create DataFrame
        dates = pd.date_range(
            start=pd.to_datetime(hourly_aq.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly_aq.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly_aq.Interval()),
            inclusive="left"
        )
        
        df = pd.DataFrame({
            "timestamp": dates,
            "pm2_5": hourly_aq.Variables(0).ValuesAsNumpy(),
            "pm10": hourly_aq.Variables(1).ValuesAsNumpy(),
            "no2": hourly_aq.Variables(2).ValuesAsNumpy(),
            "temp": hourly_w.Variables(0).ValuesAsNumpy(),
            "humidity": hourly_w.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly_w.Variables(2).ValuesAsNumpy()
        })
        
        # 4. Filter for ONLY the current/latest hour
        # We grab the last row that isn't in the future
        now = pd.Timestamp.now(tz='UTC')
        current_data = df[df['timestamp'] <= now].iloc[-1:]
        
        current_data["hour"] = current_data["timestamp"].dt.hour
        
        # 5. Push to MongoDB (Append Mode)
        if not current_data.empty:
            record = current_data.to_dict("records")[0]
            
            # Upsert: Update if exists, Insert if new (prevents duplicates)
            collection.update_one(
                {"timestamp": record["timestamp"]}, 
                {"$set": record}, 
                upsert=True
            )
            print(f"✅ Successfully added data for: {record['timestamp']}")
        else:
            print("⚠️ No valid past data found in response.")

    except Exception as e:
        print(f"❌ Error in feature pipeline: {e}")

if __name__ == "__main__":
    fetch_latest_hour()