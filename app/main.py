import os
import datetime
import pickle
import numpy as np
import certifi
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# --- SETUP ---
st.set_page_config(page_title="AQI Monitor & Forecaster", layout="wide")

MONGO_URI = os.getenv("MONGO_URI")

def get_database():
    if not MONGO_URI:
        st.error("Missing MONGO_URI in .env")
        st.stop()
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    return client.get_database("aqi_project_db")

# --- CACHED FUNCTIONS ---

@st.cache_resource
def load_model():
    """Load the trained machine learning model."""
    if not os.path.exists("aqi_model.pkl"):
        return None
    with open("aqi_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=300)
def load_data():
    """Load most recent 24 hours of data."""
    db = get_database()
    collection = db.get_collection("features")
    
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=26)

    cursor = collection.find({"timestamp": {"$gte": cutoff}}).sort("timestamp", 1)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
            
    return df.sort_values("timestamp")

def get_model_registry_data():
    """Fetch the latest model tournament results from MongoDB."""
    db = get_database()
    # Get the most recent experiment
    doc = db.model_registry.find_one(sort=[("experiment_date", -1)])
    return doc

# --- MAIN APP ---

def main():
    st.title("☁️ AQI Monitor & Forecaster")
    
    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Live Dashboard", "🤖 Model Registry"])

    # ---------------------------------------------------------
    # TAB 1: Live Dashboard (The User View)
    # ---------------------------------------------------------
    with tab1:
        df = load_data()
        model = load_model()

        if df.empty:
            st.info("Waiting for data pipeline to run...")
            return

        latest = df.iloc[-1]
        last_time = latest["timestamp"]
        
        # Prediction Logic
        predicted_pm25 = None
        if model:
            try:
                next_hour = (last_time.hour + 1) % 24
                # Features must match training order: 
                # [pm2_5, pm10, no2, temp, humidity, wind_speed, hour]
                input_features = [
                    latest.get("pm2_5", 0), latest.get("pm10", 0), latest.get("no2", 0),
                    latest.get("temp", 0), latest.get("humidity", 0), latest.get("wind_speed", 0),
                    next_hour
                ]
                predicted_pm25 = model.predict([input_features])[0]
            except Exception as e:
                st.error(f"Prediction Error: {e}")

        # Big Forecast Card
        st.markdown("### 🔮 Next Hour Forecast")
        col_pred, col_status = st.columns([1, 3])
        
        with col_pred:
            if predicted_pm25:
                delta = predicted_pm25 - latest["pm2_5"]
                st.metric(
                    label="Predicted PM2.5",
                    value=f"{predicted_pm25:.1f}",
                    delta=f"{delta:.1f}",
                    delta_color="inverse"
                )
            else:
                st.warning("Model missing.")

        with col_status:
            if predicted_pm25:
                if predicted_pm25 > 35:
                    st.error(f"⚠️ Warning: Air quality expected to be UNHEALTHY at {next_hour}:00.")
                else:
                    st.success(f"✅ Good News: Air quality expected to remain GOOD at {next_hour}:00.")

        st.divider()

        # Current Stats
        st.subheader(f"📍 Current Status (Karachi) - {last_time.strftime('%H:%M')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current PM2.5", f"{latest['pm2_5']:.1f}")
        m2.metric("Temp", f"{latest['temp']:.1f} °C")
        m3.metric("Humidity", f"{latest['humidity']:.0f}%")
        m4.metric("Wind", f"{latest['wind_speed']:.1f} km/h")

        # Charts
        left, right = st.columns(2)
        with left:
            st.markdown("#### Pollution Trend (24h)")
            fig = px.line(df, x="timestamp", y="pm2_5", markers=True)
            if predicted_pm25:
                future_time = last_time + datetime.timedelta(hours=1)
                fig.add_scatter(x=[future_time], y=[predicted_pm25], 
                            mode='markers', name='Forecast', 
                            marker=dict(color='red', size=12, symbol='star'))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Pollutant Correlations")
            corr_cols = ["pm2_5", "pm10", "no2", "temp", "humidity"]
            corr = df[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: Model Registry (The Admin/Developer View)
    # ---------------------------------------------------------
    with tab2:
        st.header("🤖 AI Model Registry")
        st.markdown("This section tracks the performance of different ML algorithms.")
        
        registry_doc = get_model_registry_data()
        
        if not registry_doc:
            st.warning("No model registry data found. Run pipelines/training_pipeline.py first.")
        else:
            # Winner Banner
            winner = registry_doc['winner']
            error = registry_doc['winner_mae']
            date = registry_doc['experiment_date']
            
            st.success(f"🏆 Current Champion: **{winner}** (Error: ±{error:.2f})")
            st.caption(f"Last trained: {date}")
            
            # Prepare data for Chart
            candidates = registry_doc['candidates']
            perf_df = pd.DataFrame(candidates)
            
            # Bar Chart of MAE (Lower is better)
            st.subheader("🥊 Model Tournament Results")
            fig = px.bar(
                perf_df, 
                x="model_name", 
                y="mae", 
                color="model_name",
                title="Model Error Comparison (Lower is Better)",
                labels={"mae": "Mean Absolute Error (PM2.5)", "model_name": "Algorithm"},
                text_auto=".2f"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Table
            st.subheader("📋 Detailed Metrics")
            st.dataframe(perf_df[["model_name", "mae", "trained_at"]])

if __name__ == "__main__":
    main()