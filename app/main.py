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
    # Get metrics safely
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

        # 🔥 REQUIRED: remove timezone for Altair
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df["hour"] = df["timestamp"].dt.hour
        return df
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

# --- MAIN UI ---
st.set_page_config(page_title="AQI Forecaster", page_icon="🌤️")

st.title("🌤️ AI-Powered Air Quality Forecaster")
st.markdown("Predicting **PM2.5** and **AQI** for Karachi (Next 72 Hours)")

col1, col2 = st.columns(2)

model, model_name, metrics = load_model()

if model:
    col1.success(f"🧠 Model Loaded: {model_name}")
    # --- NEW: Show Metrics in Expander ---
    if metrics:
        with st.expander("📊 Model Accuracy Stats"):
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{metrics.get('r2', 0)*100:.1f}%")
            m2.metric("MAE", f"{metrics.get('mae', 0):.2f}")
            m3.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
else:
    col1.error("⚠️ No Model Found!")
    st.stop()

with st.spinner("Fetching Weather Forecast..."):
    forecast_df = get_weather_forecast()
    if not forecast_df.empty:
        col2.info("🌍 Weather Forecast Fetched")
    else:
        st.stop()

# ---------------- PREDICTION ----------------
X_future = forecast_df[["temp", "humidity", "wind_speed", "hour"]]
forecast_df["Predicted_PM25"] = model.predict(X_future).astype(float)
# Calculate AQI as well
forecast_df["Predicted_AQI"] = forecast_df["Predicted_PM25"].apply(calculate_aqi)

st.divider()
st.markdown("### 📈 3-Day Pollution Forecast")

# --- DUAL TABS ---
tab1, tab2 = st.tabs(["AQI Score", "Raw PM2.5"])

with tab1:
    st.markdown("##### Air Quality Index (0-500)")
    # New AQI Chart (Area)
    chart_aqi = alt.Chart(forecast_df).mark_area(
        line={'color':'#FF4B4B'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#FF4B4B', offset=0),
                   alt.GradientStop(color='white', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X("timestamp:T", title="Time (Next 72 Hours)"),
        y=alt.Y("Predicted_AQI:Q", title="AQI Score"),
        tooltip=["timestamp", "Predicted_AQI"]
    ).properties(height=400)
    st.altair_chart(chart_aqi, use_container_width=True)
    st.caption("AQI Scale: 0-50 (Good) | 51-100 (Moderate) | 100+ (Unhealthy)")

with tab2:
    st.markdown("##### PM2.5 Concentration (µg/m³)")
    
    # --- YOUR ORIGINAL WORKING GRAPH CODE ---
    chart = alt.Chart(forecast_df).mark_line(
        point=True,
        strokeWidth=3,
        color="#FF4B4B"
    ).encode(
        x=alt.X("timestamp:T", title="Time (Next 72 Hours)"),
        y=alt.Y(
            "Predicted_PM25:Q",
            title="PM2.5 Level",
            scale=alt.Scale(
                domain=[
                    forecast_df["Predicted_PM25"].min() - 5,
                    forecast_df["Predicted_PM25"].max() + 5
                ]
            )
        ),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time"),
            alt.Tooltip("Predicted_PM25:Q", title="PM2.5", format=".1f")
        ]
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)
    # --- END OF ORIGINAL CODE ---

# ---------------- TABLE ----------------
st.markdown("### 📋 Detailed Forecast Data")
st.dataframe(
    forecast_df[["timestamp", "Predicted_AQI", "Predicted_PM25", "temp", "wind_speed"]]
)

# ---------------- METRICS ----------------
avg_aqi = forecast_df["Predicted_AQI"].mean()
max_aqi = forecast_df["Predicted_AQI"].max()

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Avg AQI", f"{avg_aqi:.0f}")
c2.metric("Max AQI", f"{max_aqi:.0f}")

if avg_aqi <= 50:
    c3.success("Status: Good 🌳")
elif avg_aqi <= 100:
    c3.warning("Status: Moderate 😐")
else:
    c3.error("Status: Unhealthy 😷")