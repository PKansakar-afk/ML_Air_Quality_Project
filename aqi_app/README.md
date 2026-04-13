# AQI Intelligence — Flask App

## Setup

### Step 1 — Export models from Jupyter

Copy `save_models_cell.py` content into a new cell **at the end of your notebook**
and run it. This creates:

```text
aqi_app/
  models/  rf_model.pkl  xgb_model.pkl  lgbm_model.pkl  ridge_model.pkl  feature_columns.pkl
  data/    historical_stats.pkl
```

### Step 2 — Get a free WAQI token

Visit <https://aqicn.org/data-platform/token/> and register (takes 30 seconds).

### Step 3 — Install dependencies

```bash
cd aqi_app
pip install -r requirements.txt
```

### Step 4 — Set your WAQI token and run

**Windows:**

```cmd

set WAQI_TOKEN=your_token_here
python app.py

```

**Mac/Linux:**

```bash

export WAQI_TOKEN=your_token_here
python app.py
```

Open <http://localhost:5000>

---

## API Endpoints

| Method | Route | Description |
| -------- | ------- | ------------- |
| GET | `/` | Main dashboard |
| POST | `/api/nowcast` | Real-time prediction `{"city":"Delhi"}` |
| GET | `/api/historical?city=Delhi&month=1&day=15` | Historical stats |
| GET | `/api/heatmap?city=Delhi` | Monthly risk data |

---

## How Nowcasting Works

1. **WAQI API** → fetches PM2.5, PM10, NO₂, SO₂, CO for the selected city
2. **Open-Meteo API** → fetches temperature, humidity, wind speed (no key needed)
3. **Lag features** → current reading used as lag-1; city/month historical medians
   used for lag-3, lag-7, and rolling statistics
4. All 4 tuned pipelines (RF, XGBoost, LightGBM, Ridge) predict independently

> Note: If WAQI doesn't have data for a smaller city (e.g. Jorapokhar),
> predictions fall back to historical medians with a warning shown in the UI.

---

## File Structure

```text
aqi_app/
├── app.py                  Flask backend
├── save_models_cell.py     Run in Jupyter to export pkl files
├── requirements.txt
├── models/                 Auto-created by save_models_cell.py
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── ridge_model.pkl
│   └── feature_columns.pkl
├── data/
│   └── historical_stats.pkl
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/main.js
```
