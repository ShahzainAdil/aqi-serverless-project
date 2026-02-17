# 🌤️ AI-Powered Air Quality Forecaster (Serverless MLOps)

![Status](https://img.shields.io/badge/Status-Live-success)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pipeline](https://img.shields.io/badge/Pipeline-Automated-orange)

## 🚀 Live Demo
**[Click here to launch the Dashboard](https://aqi-serverless-project-csvcfw6mjcsappwnxkupxds.streamlit.app/)**

## 📖 Project Overview
This project is an **End-to-End Serverless Machine Learning Pipeline** that predicts Air Quality (AQI & PM2.5) in Karachi for the next 72 hours. Unlike static notebooks, this system runs autonomously in the cloud, utilizing a robust feature engineering pipeline, automated retraining, and a real-time dashboard with historical analysis.

### 🏗️ Architecture & Features

#### 1. 🤖 Feature Pipeline (Data Engineering)
* **Automated Collection:** Fetches raw weather and pollutant data from **Open-Meteo APIs** every hour.
* **Feature Engineering:** Computes time-based features and derives critical metrics like wind-speed interactions.
* **Storage:** Stores processed features in a **MongoDB Atlas** Feature Store.
* **Historical Backfill:** Includes a specialized script to fetch and process past data (30+ days) for robust training.

#### 2. 🧠 Training Pipeline (AutoML)
* **Daily Retraining:** Wakes up every **24 hours** via GitHub Actions to retrain models on the latest data.
* **Model Tournament:** Automatically trains and compares multiple algorithms (Random Forest, Gradient Boosting, Linear Regression).
* **Dynamic Evaluation:** The system automatically promotes the model with the best RMSE score to "Champion" status.

#### 3. 📊 Dashboard & Analytics
* **Live Forecast:** Visualizes AQI predictions for the next 3 days.
* **Exploratory Data Analysis (EDA):** A dedicated tab for analyzing historical pollution trends and correlations (e.g., Wind Speed vs. PM2.5).

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Database:** MongoDB Atlas (Cloud NoSQL)
* **Orchestration:** GitHub Actions (CI/CD & Cron Jobs)
* **Frontend:** Streamlit Cloud

## 📈 Model Performance (Benchmark)
*Current Champion: Linear Regressor*

| Metric | Typical Score | Description |
| :--- | :--- | :--- |
| **R² Score** | **~80.0%** | Explains 80% of the variance in pollution levels. |
| **MAE** | **±3.22** | Average error in PM2.5 units. |
| **RMSE** | **±5.0** | Root Mean Square Error. |

*> **Note:** Since this system retrains daily on new real-world data, these metrics may fluctuate slightly over time.*

---
*Built by Shahzain*
