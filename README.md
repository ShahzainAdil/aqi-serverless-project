# 🌤️ AI-Powered Air Quality Forecaster (Serverless MLOps)

![Status](https://img.shields.io/badge/Status-Live-success)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-92%25-green)

## 🚀 Live Demo
**[Click here to launch the Dashboard](https://aqi-serverless-project-csvcfw6mjcsappwnxkupxds.streamlit.app/)**

## 📖 Project Overview
This project is an **End-to-End Serverless Machine Learning Pipeline** that predicts Air Quality (AQI & PM2.5) in Karachi for the next 72 hours. Unlike static notebooks, this system runs autonomously in the cloud, fetching live data, retraining models, and updating the dashboard without manual intervention.

### 🏗️ Serverless Architecture
The system follows a modern MLOps architecture using **GitHub Actions** and **MongoDB Atlas**:

1.  **🤖 Data Collector (Hourly Robot):** * Wakes up every **hour** (via GitHub Actions cron).
    * Fetches real-time weather & pollution data from Open-Meteo APIs.
    * Updates the **MongoDB Atlas** feature store.
2.  **🧠 Model Trainer (Daily Robot):** * Wakes up every **24 hours**.
    * Retrains 3 models (Random Forest, XGBoost, Linear Reg).
    * Evaluates them using **RMSE** and **R² Score**.
    * Saves the "Champion Model" to the registry.
3.  **📊 User Interface (Streamlit):** * A live dashboard that loads the champion model.
    * Visualizes forecasts using interactive Altair charts.
    * Converts raw PM2.5 data into human-readable **AQI Scores**.

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Database:** MongoDB Atlas (Cloud NoSQL)
* **Orchestration:** GitHub Actions (CI/CD & Cron Jobs)
* **Frontend:** Streamlit Cloud

## 📈 Model Performance
The current champion model (**Random Forest**) achieves:
* **R² Score:** 92.0%
* **MAE:** ±2.63 µg/m³
* **RMSE:** ±3.59 µg/m³

---
*Built by Shahzain*