import streamlit as st
import pandas as pd
import pymongo
import os
import pickle
import requests
import certifi
import altair as alt
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()
# Try to get URI from local .env, otherwise from Streamlit Secrets
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

# 2. Connect to Database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())

client = init_connection()
db = client["aqi_project_db"]

# --- HELPER: Convert PM2.5 to AQI (US EPA Standard) ---
def calculate_aqi(pm25):
    if pm25 <= 12.0:
        return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    else:
        return 300 # Hazardous

# 3. Helper: Load the Trained Model & Metrics
def load_model():
    model_doc = db["model_registry"].find_one(sort=[("timestamp", -1)])
    if not model_doc:
        return None, None, {}
    
    model = pickle.loads(model_doc["model_binary"])
    metrics = model_doc.get("metrics", {})
    return model, model_doc["model_name"], metrics

# 4. Helper: Fetch Future Weather
def get_weather_forecast():
    LAT, LON = 24.8607, 67.0011
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
        "forecast_days": 3
    }
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temp": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"]
        })
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df["hour"] = df["timestamp"].dt.hour
        return df
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

# 5. Helper: Fetch Historical Data (For EDA)
@st.cache_data(ttl=3600) # Cache for 1 hour to stay fast
def get_historical_data():
    # Fetch last 500 records
    cursor = db["pollution_data"].find().sort("timestamp", -1).limit(500)
    data = list(cursor)
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# --- MAIN UI ---
st.set_page_config(page_title="AQI Forecaster", page_icon="🌤️", layout="wide")

st.title("🌤️ AI-Powered Air Quality Forecaster")
st.markdown("Predicting **PM2.5** and **AQI** for Karachi (Next 72 Hours)")

# --- LOAD ASSETS ---
model, model_name, metrics = load_model()

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("Project Info")
    st.markdown("""
    **Serverless MLOps Pipeline**
    - 🤖 **Collector:** GitHub Actions (Hourly)
    - 🧠 **Trainer:** Auto-Retrains Daily
    - ☁️ **Store:** MongoDB Atlas
    """)
    if model:
        st.success(f"🧠 Active Model: {model_name}")
        st.metric("Model Accuracy (R²)", f"{metrics.get('r2', 0)*100:.1f}%")

# --- MAIN TABS ---
tab_forecast, tab_eda = st.tabs(["🔮 Live Forecast", "📊 Historical Analysis (EDA)"])

# ==========================
# TAB 1: LIVE FORECAST
# ==========================
with tab_forecast:
    col1, col2 = st.columns(2)
    
    with st.spinner("Fetching Weather Forecast..."):
        forecast_df = get_weather_forecast()

    if not forecast_df.empty and model:
        # Prediction
        X_future = forecast_df[["temp", "humidity", "wind_speed", "hour"]]
        forecast_df["Predicted_PM25"] = model.predict(X_future).astype(float)
        forecast_df["Predicted_AQI"] = forecast_df["Predicted_PM25"].apply(calculate_aqi)

        # 1. AQI Area Chart
        st.subheader("📈 72-Hour AQI Forecast")
        chart_aqi = alt.Chart(forecast_df).mark_area(
            line={'color':'#FF4B4B'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#FF4B4B', offset=0),
                       alt.GradientStop(color='white', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("Predicted_AQI:Q", title="Predicted AQI"),
            tooltip=["timestamp", "Predicted_AQI", "temp", "wind_speed"]
        ).properties(height=350)
        st.altair_chart(chart_aqi, use_container_width=True)

        # 2. Metrics
        avg_aqi = forecast_df["Predicted_AQI"].mean()
        max_aqi = forecast_df["Predicted_AQI"].max()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Average AQI", f"{avg_aqi:.0f}")
        c2.metric("Worst Peak AQI", f"{max_aqi:.0f}")
        
        if avg_aqi <= 50:
            c3.success("Overall Status: Good 🌳")
        elif avg_aqi <= 100:
            c3.warning("Overall Status: Moderate 😐")
        else:
            c3.error("Overall Status: Unhealthy 😷")

        # 3. Data Table
        with st.expander("📋 View Raw Forecast Data"):
            st.dataframe(forecast_df)
    else:
        st.warning("⚠️ Waiting for data or model...")

# ==========================
# TAB 2: HISTORICAL EDA
# ==========================
with tab_eda:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Analyzing the last **500 hours** of collected sensor data.")
    
    historical_df = get_historical_data()
    
    if not historical_df.empty:
        # 1. Trend Line
        st.subheader("1. Pollution Trend (PM2.5)")
        chart_trend = alt.Chart(historical_df).mark_line(color='orange').encode(
            x=alt.X('timestamp:T', title='Date & Time'),
            y=alt.Y('pm2_5:Q', title='Recorded PM2.5'),
            tooltip=['timestamp', 'pm2_5', 'pm10']
        ).properties(height=300)
        st.altair_chart(chart_trend, use_container_width=True)

        # 2. Correlation Scatter (Wind vs PM2.5)
        st.subheader("2. Correlation: Wind Speed vs. Pollution")
        
        
        chart_corr = alt.Chart(historical_df).mark_circle(size=60).encode(
            x=alt.X('wind_speed:Q', title='Wind Speed (km/h)'),
            y=alt.Y('pm2_5:Q', title='PM2.5 Level'),
            color=alt.Color('pm2_5', scale=alt.Scale(scheme='magma')),
            tooltip=['wind_speed', 'pm2_5', 'temp']
        ).properties(height=350)
        st.altair_chart(chart_corr, use_container_width=True)
        
        # 3. Statistics
        st.subheader("3. Dataset Statistics")
        st.write(historical_df[["pm2_5", "pm10", "temp", "wind_speed", "humidity"]].describe())
        
    else:
        st.info("No historical data found in database yet. Wait for the collector to run!")
