# AQI-Serverless-Project

A minimal scaffold for an Air Quality Index (AQI) serverless project.

## Structure
- `app/` — Streamlit dashboard
- `pipelines/` — ETL and training logic
- `notebooks/` — EDA and experiments
- `model/` — local model artifacts (optional)
- `.github/workflows/` — CI/CD automation

## Quick start
1. Create a virtualenv and install requirements: `pip install -r requirements.txt`
2. Run dashboard: `streamlit run app/main.py` (create `app/main.py` when ready)
