import os
import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise SystemExit("MONGO_URI not found in environment")

LAT = 24.8607
LON = 67.0011

def get_collection():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.get_database("aqi_project_db")
    return db.get_collection("features")

def fetch_weather(start_date: str, end_date: str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hourly", {})

def fetch_air_quality(start_date: str, end_date: str):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,nitrogen_dioxide",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hourly", {})

def build_records(weather_hourly: dict, air_hourly: dict):
    times = weather_hourly.get("time", [])
    # Build mapping for air times -> index
    air_time_index = {t: i for i, t in enumerate(air_hourly.get("time", []))}
    records = []
    for i, t in enumerate(times):
        try:
            temp = weather_hourly.get("temperature_2m", [None])[i]
            humidity = weather_hourly.get("relative_humidity_2m", [None])[i]
            wind_speed = weather_hourly.get("wind_speed_10m", [None])[i]

            air_idx = air_time_index.get(t)
            if air_idx is None:
                continue
            pm2_5 = air_hourly.get("pm2_5", [None])[air_idx]
            pm10 = air_hourly.get("pm10", [None])[air_idx]
            no2 = air_hourly.get("nitrogen_dioxide", [None])[air_idx]

            # Skip if any value is None / NaN
            values = [pm2_5, pm10, no2, temp, humidity, wind_speed]
            if any(pd.isna(v) for v in values):
                continue

            ts = pd.to_datetime(t, utc=True).to_pydatetime()
            rec = {
                "timestamp": ts,
                "pm2_5": float(pm2_5),
                "pm10": float(pm10),
                "no2": float(no2),
                "temp": float(temp),
                "humidity": float(humidity),
                "wind_speed": float(wind_speed),
            }
            records.append(rec)
        except (IndexError, TypeError, ValueError):
            continue
    return records

def main():
    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()

    weather_hourly = fetch_weather(start_date, end_date)
    air_hourly = fetch_air_quality(start_date, end_date)

    records = build_records(weather_hourly, air_hourly)
    if not records:
        print("No valid records found to insert.")
        return

    coll = get_collection()
    result = coll.insert_many(records)
    print(f"✅ Successfully inserted {len(result.inserted_ids)} rows of real history")

if __name__ == "__main__":
    main()