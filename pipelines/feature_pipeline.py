import os
import sys
import datetime
from typing import Dict, Any, Optional

import certifi
import requests
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY_LAT = os.getenv("CITY_LAT")
CITY_LON = os.getenv("CITY_LON")


if not all([MONGO_URI, OPENWEATHER_API_KEY, CITY_LAT, CITY_LON]):
    missing = [k for k, v in (
        ("MONGO_URI", MONGO_URI),
        ("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY),
        ("CITY_LAT", CITY_LAT),
        ("CITY_LON", CITY_LON),
    ) if not v]
    print(f"Missing environment variables: {', '.join(missing)}")
    sys.exit(1)


def get_mongo_collection():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client.get_database("aqi_project_db")
    return db.get_collection("features")


def fetch_data(api_key: str, lat: str, lon: str, timeout: int = 10) -> Dict[str, Any]:
    """Call OpenWeather Current Weather and Air Pollution endpoints and return parsed data."""
    base = "https://api.openweathermap.org/data/2.5"

    # Current weather
    weather_url = f"{base}/weather"
    params_w = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    r_w = requests.get(weather_url, params=params_w, timeout=timeout)
    r_w.raise_for_status()
    w = r_w.json()

    # Air pollution
    pollution_url = f"{base}/air_pollution"
    params_p = {"lat": lat, "lon": lon, "appid": api_key}
    r_p = requests.get(pollution_url, params=params_p, timeout=timeout)
    r_p.raise_for_status()
    p = r_p.json()

    return {"weather": w, "pollution": p}


def transform_data(raw: Dict[str, Any], ts: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """Combine API responses into a single document with derived features and timestamp."""
    if ts is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc)

    weather = raw.get("weather", {})
    pollution = raw.get("pollution", {})

    temp = None
    humidity = None
    wind_speed = None
    try:
        main = weather.get("main", {})
        temp = main.get("temp")
        humidity = main.get("humidity")
        wind_speed = (weather.get("wind") or {}).get("speed")
    except Exception:
        pass

    pm2_5 = None
    pm10 = None
    no2 = None
    aqi = None
    try:
        # OpenWeather returns a list under 'list' for air pollution
        plist = pollution.get("list", [])
        if plist:
            comps = plist[0].get("components", {})
            pm2_5 = comps.get("pm2_5")
            pm10 = comps.get("pm10")
            no2 = comps.get("no2")
            aqi = (plist[0].get("main") or {}).get("aqi")
    except Exception:
        pass

    ts_hour = ts.replace(minute=0, second=0, microsecond=0)

    doc = {
        "timestamp": ts,
        "timestamp_hour": ts_hour,
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pm2_5": pm2_5,
        "pm10": pm10,
        "no2": no2,
        "aqi": aqi,
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": ts.weekday() >= 5,
        "raw": raw,
    }

    return doc


def load_data(doc: Dict[str, Any], collection) -> None:
    """Insert document into MongoDB if a document with same hour doesn't exist."""
    existing = collection.find_one({"timestamp_hour": doc["timestamp_hour"]})
    if existing:
        print(f"ℹ️ Data for {doc['timestamp_hour'].isoformat()} already exists. Skipping insert.")
        return

    res = collection.insert_one(doc)
    print(f"✅ Data for {doc['timestamp'].isoformat()} inserted successfully (id={res.inserted_id})")


def main():
    collection = get_mongo_collection()
    raw = fetch_data(OPENWEATHER_API_KEY, CITY_LAT, CITY_LON)
    doc = transform_data(raw)
    load_data(doc, collection)


if __name__ == "__main__":
    main()
