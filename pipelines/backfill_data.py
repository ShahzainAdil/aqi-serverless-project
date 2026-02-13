import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import pymongo
import os
import certifi
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 1. Load Secrets
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# 2. Setup MongoDB
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aqi_project_db"]
collection = db["pollution_data"]

# 3. Setup Open-Meteo API Client with Cache
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def backfill_history():
    print("📡 Connecting to Open-Meteo for historical data (Last 30 Days)...")
    
    # Coordinates for Karachi
    LAT = 24.8607
    LON = 67.0011
    
    # Calculate Date Range (Last 30 Days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["pm2_5", "pm10"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto"
    }
    
    # Get Air Quality Data
    aq_responses = openmeteo.weather_api(url, params=params)
    response = aq_responses[0]

    # Process Hourly Data
    hourly = response.Hourly()
    pm25 = hourly.Variables(0).ValuesAsNumpy()
    pm10 = hourly.Variables(1).ValuesAsNumpy()
    
    # --- FETCH WEATHER DATA (Temp, Humidity, Wind) ---
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto"
    }
    
    weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
    w_response = weather_responses[0]
    w_hourly = w_response.Hourly()
    
    temp = w_hourly.Variables(0).ValuesAsNumpy()
    humidity = w_hourly.Variables(1).ValuesAsNumpy()
    wind_speed = w_hourly.Variables(2).ValuesAsNumpy()

    # --- 🛠️ BULLETPROOF ALIGNMENT FIX ---
    # We use the length of the 'pm25' array to define how many timestamps we need.
    # This prevents the "Arrays must be same length" error.
    
    # 1. Generate Timestamps
    start_ts = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    interval = hourly.Interval()
    n_points = len(pm25) # Trust the data length
    
    timestamps = pd.date_range(start=start_ts, periods=n_points, freq=f"{interval}s")

    # 2. Create DataFrame safely
    # Note: We slice all arrays [:n_points] just in case weather api returns 1 extra hour
    df = pd.DataFrame({
        "timestamp": timestamps,
        "pm2_5": pm25,
        "pm10": pm10,
        "temp": temp[:n_points],
        "humidity": humidity[:n_points],
        "wind_speed": wind_speed[:n_points]
    })
    
    # 3. Add derived features
    df["timestamp"] = df["timestamp"].dt.tz_localize(None) # Remove timezone for Mongo
    df["hour"] = df["timestamp"].dt.hour
    
    # 4. Upload to MongoDB
    data_dict = df.to_dict("records")
    
    if data_dict:
        # Use bulk write to avoid duplicates
        operations = [
            pymongo.UpdateOne(
                {"timestamp": record["timestamp"]}, 
                {"$set": record}, 
                upsert=True
            )
            for record in data_dict
        ]
        result = collection.bulk_write(operations)
        print(f"✅ Backfill Complete: {len(data_dict)} records processed!")
        print(f"   (Inserted/Updated: {result.upserted_count + result.modified_count})")
    else:
        print("⚠️ No data found to insert.")

if __name__ == "__main__":
    try:
        backfill_history()
    except Exception as e:
        print(f"❌ Error in backfill: {e}")