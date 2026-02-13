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
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI and "MONGO_URI" in st.secrets:
    MONGO_URI = st.secrets["MONGO_URI"]

# 2. Connect to Database
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())

client = init_connection()
db = client["aqi_project_db"]

# --- HELPERS ---
def calculate_aqi(pm25):
    if pm25 <= 12.0: return ((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0
    elif pm25 <= 35.4: return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4: return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4: return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    else: return 300

def load_model_data():
    model_doc = db["model_registry"].find_one(sort=[("timestamp", -1)])
    if not model_doc:
        return None, None, {}, []
    
    model = pickle.loads(model_doc["model_binary"])
    return model, model_doc["model_name"], model_doc.get("metrics", {}), model_doc.get("leaderboard", [])

def get_latest_actual_pm25():
    """Fetches the most recent actual PM2.5 reading from DB to start the forecast loop."""
    latest = db["pollution_data"].find_one(sort=[("timestamp", -1)])
    if latest:
        return latest.get("pm2_5", 50.0) # Default to 50 if missing
    return 50.0 

def get_weather_forecast():
    LAT, LON = 24.8607, 67.0011
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": LAT, "longitude": LON, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m", "forecast_days": 3}
    try:
        resp = requests.get(url, params=params).json()
        hourly = resp["hourly"]
        df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"]), "temp": hourly["temperature_2m"], "humidity": hourly["relative_humidity_2m"], "wind_speed": hourly["wind_speed_10m"]})
        df["hour"] = df["timestamp"].dt.hour
        return df
    except: return pd.DataFrame()

def get_historical_data():
    cursor = db["pollution_data"].find().sort("timestamp", -1).limit(500)
    df = pd.DataFrame(list(cursor))
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# --- UI SETUP ---
st.set_page_config(page_title="AQI Forecaster", page_icon="🌤️", layout="wide")
st.title("🌤️ AI-Powered Air Quality Forecaster")

# Load Everything
model, model_name, metrics, leaderboard = load_model_data()

# Sidebar
with st.sidebar:
    st.header("🤖 Model Status")
    if model:
        st.success(f"Active: {model_name}")
        st.metric("R² Score", f"{metrics.get('r2', 0)*100:.1f}%")
        st.metric("Error (MAE)", f"{metrics.get('mae', 0):.2f}")
    else:
        st.error("No model trained yet!")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Live Forecast", "📊 Historical Analysis", "🧠 Model Tournament"])

# --- TAB 1: FORECAST ---
with tab1:
    forecast_df = get_weather_forecast()
    
    if not forecast_df.empty and model:
        # --- 🔁 RECURSIVE FORECASTING LOOP ---
        current_lag = get_latest_actual_pm25()
        predictions = []
        
        for index, row in forecast_df.iterrows():
            input_data = pd.DataFrame({
                "temp": [row["temp"]],
                "humidity": [row["humidity"]],
                "wind_speed": [row["wind_speed"]],
                "hour": [row["hour"]],
                "pm2_5_lag1": [current_lag] 
            })
            pred = model.predict(input_data)[0]
            predictions.append(pred)
            current_lag = pred # Update lag for next loop
            
        forecast_df["Predicted_PM25"] = predictions
        forecast_df["Predicted_AQI"] = forecast_df["Predicted_PM25"].apply(calculate_aqi)
        
        st.altair_chart(alt.Chart(forecast_df).mark_area(line={'color':'#FF4B4B'}, color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#FF4B4B', offset=0), alt.GradientStop(color='white', offset=1)], x1=1, x2=1, y1=1, y2=0)).encode(x='timestamp:T', y='Predicted_AQI:Q').properties(height=300), use_container_width=True)
        
        avg_aqi = forecast_df["Predicted_AQI"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Forecast AQI", f"{avg_aqi:.0f}")
        c2.metric("Peak AQI", f"{forecast_df['Predicted_AQI'].max():.0f}")
        status = "Good 🌳" if avg_aqi <= 50 else "Moderate 😐" if avg_aqi <= 100 else "Unhealthy 😷"
        c3.info(f"Status: {status}")

# --- TAB 2: HISTORY ---
with tab2:
    hist_df = get_historical_data()
    if not hist_df.empty:
        st.subheader("Pollution Trend (Last 30 Days)")
        
        # 1. Line Chart (Trend)
        chart_trend = alt.Chart(hist_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'), 
            y=alt.Y('pm2_5:Q', title='PM2.5')
        ).properties(height=300)
        
        st.altair_chart(chart_trend, use_container_width=True)
        
        st.subheader("Correlations")
        
        # 2. Scatter Chart (Wind vs Pollution)
        chart_corr = alt.Chart(hist_df).mark_circle().encode(
            x=alt.X('wind_speed:Q', title='Wind Speed'), 
            y=alt.Y('pm2_5:Q', title='PM2.5'), 
            color='temp:Q'
        ).properties(height=300)
        
        st.altair_chart(chart_corr, use_container_width=True)