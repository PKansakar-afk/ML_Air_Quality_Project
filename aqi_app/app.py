"""
AQI Prediction Flask App
Run: python app.py
Set env var WAQI_TOKEN before running (free at aqicn.org/data-platform/token/)
"""
import os
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import date
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load models & stats ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

def _pkl(fname, folder=MODELS_DIR):
    with open(os.path.join(folder, fname), "rb") as f:
        return pickle.load(f)

print("Loading models...", end=" ", flush=True)
MODELS = {
    "Random Forest": _pkl("rf_model.pkl"),
    "XGBoost":       _pkl("xgb_model.pkl"),
    "LightGBM":      _pkl("lgbm_model.pkl"),
    "Ridge":         _pkl("ridge_model.pkl"),
}
FEATURE_COLUMNS = _pkl("feature_columns.pkl")
STATS           = _pkl("historical_stats.pkl", folder=DATA_DIR)
print("done")

MONTHLY_STATS      = STATS["monthly_stats"]
DAILY_STATS        = STATS["daily_stats"]
BUCKET_DIST        = STATS["bucket_dist"]
CITY_MONTH_MEDIANS = STATS["city_month_medians"]
CITIES             = STATS["cities"]

# ── API keys & coords ──────────────────────────────────────────────────
WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "demo")  # demo token has rate limits

CITY_COORDS = {
    "Ahmedabad":          (23.0225, 72.5714),
    "Aizawl":             (23.7271, 92.7176),
    "Amaravati":          (16.5131, 80.5165),
    "Amritsar":           (31.6340, 74.8723),
    "Bengaluru":          (12.9716, 77.5946),
    "Bhopal":             (23.2599, 77.4126),
    "Brajrajnagar":       (21.8216, 83.9216),
    "Chandigarh":         (30.7333, 76.7794),
    "Chennai":            (13.0827, 80.2707),
    "Coimbatore":         (11.0168, 76.9558),
    "Delhi":              (28.6139, 77.2090),
    "Ernakulam":          (9.9816,  76.2999),
    "Gurugram":           (28.4601, 77.0199),
    "Guwahati":           (26.1445, 91.7362),
    "Hyderabad":          (17.3850, 78.4867),
    "Jaipur":             (26.9124, 75.7873),
    "Jorapokhar":         (23.7004, 86.4125),
    "Kochi":              (9.9312,  76.2673),
    "Kolkata":            (22.5726, 88.3639),
    "Mumbai":             (19.0760, 72.8777),
    "Patna":              (25.5941, 85.1376),
    "Shillong":           (25.5788, 91.8933),
    "Talcher":            (20.9509, 85.2163),
    "Thiruvananthapuram": (8.5241,  76.9366),
    "Visakhapatnam":      (17.6868, 83.2185),
}

# WAQI station slugs for major cities (others fall back to city name)
WAQI_MAP = {
    "Delhi": "delhi", "Mumbai": "mumbai", "Bengaluru": "bangalore",
    "Chennai": "chennai", "Kolkata": "kolkata", "Hyderabad": "hyderabad",
    "Ahmedabad": "ahmedabad", "Chandigarh": "chandigarh", "Jaipur": "jaipur",
    "Amritsar": "amritsar", "Bhopal": "bhopal", "Patna": "patna",
    "Guwahati": "guwahati", "Visakhapatnam": "visakhapatnam",
    "Coimbatore": "coimbatore", "Kochi": "kochi", "Ernakulam": "kochi",
    "Thiruvananthapuram": "thiruvananthapuram", "Gurugram": "gurugram",
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ── Helpers ────────────────────────────────────────────────────────────
def aqi_bucket(aqi):
    if   aqi <= 50:  return ("Good",         "#00c853")
    elif aqi <= 100: return ("Satisfactory",  "#aeea00")
    elif aqi <= 200: return ("Moderate",      "#ffd600")
    elif aqi <= 300: return ("Poor",          "#ff6d00")
    elif aqi <= 400: return ("Very Poor",     "#dd2c00")
    else:            return ("Severe",        "#6a0080")


def fetch_waqi(city: str) -> tuple:
    """Call WAQI API. Returns (data_dict | None, error_str | None)."""
    station = WAQI_MAP.get(city, city.lower().replace(" ", "-"))
    url     = f"https://api.waqi.info/feed/{station}/?token={WAQI_TOKEN}"
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        if j.get("status") == "ok":
            d    = j["data"]
            iaqi = d.get("iaqi", {})
            return {
                "aqi":     d.get("aqi"),
                "pm25":    iaqi.get("pm25",  {}).get("v"),
                "pm10":    iaqi.get("pm10",  {}).get("v"),
                "no2":     iaqi.get("no2",   {}).get("v"),
                "so2":     iaqi.get("so2",   {}).get("v"),
                "co":      iaqi.get("co",    {}).get("v"),
                "station": d["city"]["name"],
                "updated": d["time"]["s"],
            }, None
        return None, j.get("data", "WAQI API returned non-ok status")
    except Exception as e:
        return None, str(e)


def fetch_weather(city: str) -> tuple:
    """Call Open-Meteo API (no key needed). Returns (data_dict | None, error | None)."""
    coords = CITY_COORDS.get(city)
    if not coords:
        return None, "Coordinates not found"
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        f"&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=10)
        c = r.json().get("current", {})
        return {
            "temp":     c.get("temperature_2m"),
            "humidity": c.get("relative_humidity_2m"),
            "wind":     c.get("wind_speed_10m"),
        }, None
    except Exception as e:
        return None, str(e)


def get_medians(city: str, month: int) -> dict:
    """Return city/month median values for lag feature construction."""
    df = CITY_MONTH_MEDIANS
    row = df[(df["City"] == city) & (df["month"] == month)]
    if row.empty:
        row = df[df["City"] == city]
    return row.iloc[0].to_dict() if not row.empty else {}


def build_feature_vector(city, target_date, waqi, weather) -> pd.DataFrame:
    """
    Construct a one-row DataFrame matching X_train's column order.

    Strategy for lag/rolling features:
      - lag1  → use current live reading (best proxy for 'yesterday')
      - lag3, lag7, rolling → use city/month historical medians
    """
    month = target_date.month
    med   = get_medians(city, month)

    def g(live_key, med_key, default=0.0):
        """live → median → default"""
        v = (waqi or {}).get(live_key)
        if v is not None: return float(v)
        v = med.get(med_key)
        if v is not None: return float(v)
        return default

    pm25    = g("pm25", "PM2.5", 50.0)
    pm10    = g("pm10", "PM10",  80.0)
    no2     = g("no2",  "NO2",   30.0)
    so2     = g("so2",  "SO2",   15.0)
    co      = g("co",   "CO",    1.5)
    no      = float(med.get("NO",  20.0))
    nox     = float(med.get("NOx", 45.0))
    aqi_est = g("aqi",  "AQI",   150.0)

    temp     = float((weather or {}).get("temp",     med.get("Temp_Mean",     25.0)))
    humidity = float((weather or {}).get("humidity", med.get("Humidity_Mean", 60.0)))
    wind     = float((weather or {}).get("wind",     med.get("Wind_Speed_Max",10.0)))

    ms = np.sin(2 * np.pi * month / 12)
    mc = np.cos(2 * np.pi * month / 12)

    def m(key, fallback): return float(med.get(key, fallback))

    row = {
        "City": city,
        # ── Base pollutants ──────────────────────────────
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
        "NOx": nox, "CO": co, "SO2": so2,
        # ── Weather ───────────────────────────────────────
        "Temp_Mean": temp, "Humidity_Mean": humidity, "Wind_Speed_Max": wind,
        # ── Date ─────────────────────────────────────────
        "year": target_date.year, "month": month,
        "day": target_date.day, "day_of_week": target_date.weekday(),
        "month_sin": ms, "month_cos": mc,
        # ── Lag 1 (current reading as proxy) ─────────────
        "PM2.5_lag1": pm25,    "PM10_lag1": pm10,   "NO_lag1": no,
        "NO2_lag1":   no2,     "NOx_lag1":  nox,    "CO_lag1": co,
        "SO2_lag1":   so2,     "AQI_lag1":  aqi_est,
        # ── Lag 3 & 7 (historical medians) ───────────────
        "PM2.5_lag3": m("PM2.5",pm25), "PM2.5_lag7": m("PM2.5",pm25),
        "PM10_lag3":  m("PM10", pm10), "PM10_lag7":  m("PM10", pm10),
        "NO_lag3":    m("NO",   no),   "NO_lag7":    m("NO",   no),
        "NO2_lag3":   m("NO2",  no2),  "NO2_lag7":   m("NO2",  no2),
        "NOx_lag3":   m("NOx",  nox),  "NOx_lag7":   m("NOx",  nox),
        "CO_lag3":    m("CO",   co),   "CO_lag7":    m("CO",   co),
        "SO2_lag3":   m("SO2",  so2),  "SO2_lag7":   m("SO2",  so2),
        "AQI_lag3":   m("AQI",  aqi_est), "AQI_lag7": m("AQI", aqi_est),
        # ── Rolling means (historical medians) ───────────
        "PM2.5_roll7_mean":  m("PM2.5",pm25), "PM2.5_roll30_mean": m("PM2.5",pm25), "PM2.5_roll7_std":  5.0,
        "PM10_roll7_mean":   m("PM10", pm10), "PM10_roll30_mean":  m("PM10", pm10), "PM10_roll7_std":   10.0,
        "NO_roll7_mean":     m("NO",   no),   "NO_roll30_mean":    m("NO",   no),   "NO_roll7_std":     5.0,
        "NO2_roll7_mean":    m("NO2",  no2),  "NO2_roll30_mean":   m("NO2",  no2),  "NO2_roll7_std":    5.0,
        "NOx_roll7_mean":    m("NOx",  nox),  "NOx_roll30_mean":   m("NOx",  nox),  "NOx_roll7_std":    5.0,
        "CO_roll7_mean":     m("CO",   co),   "CO_roll30_mean":    m("CO",   co),   "CO_roll7_std":     0.5,
        "SO2_roll7_mean":    m("SO2",  so2),  "SO2_roll30_mean":   m("SO2",  so2),  "SO2_roll7_std":    3.0,
        "AQI_roll7_mean":    m("AQI",  aqi_est), "AQI_roll30_mean": m("AQI",aqi_est), "AQI_roll7_std": 20.0,
    }

    df = pd.DataFrame([row])
    # Fill any missing columns (safety net)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]


# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", cities=CITIES)


@app.route("/api/nowcast", methods=["POST"])
def nowcast():
    body = request.get_json(force=True)
    city = body.get("city", "").strip()
    if not city or city not in CITY_COORDS:
        return jsonify({"error": f"Unknown city: '{city}'"}), 400

    today = date.today()
    month = today.month

    # ── Live data ──────────────────────────────────────────
    waqi_data,    waqi_err    = fetch_waqi(city)
    weather_data, weather_err = fetch_weather(city)

    # ── Predict ────────────────────────────────────────────
    X = build_feature_vector(city, today, waqi_data or {}, weather_data or {})
    predictions = {}
    for name, pipe in MODELS.items():
        try:
            val             = max(0.0, float(pipe.predict(X)[0]))
            bucket, color   = aqi_bucket(val)
            predictions[name] = {"aqi": round(val, 1), "bucket": bucket, "color": color}
        except Exception as e:
            predictions[name] = {"aqi": None, "error": str(e)}

    # ── Historical context (monthly) ──────────────────────
    hist_row = MONTHLY_STATS[
        (MONTHLY_STATS["City"] == city) & (MONTHLY_STATS["month"] == month)
    ]
    historical = {}
    if not hist_row.empty:
        r = hist_row.iloc[0]
        historical = {
            "mean": round(float(r["mean"]), 1),
            "min":  round(float(r["min"]),  1),
            "max":  round(float(r["max"]),  1),
        }

    # ── Bucket distribution (monthly) ────────────────────
    bd = BUCKET_DIST[
        (BUCKET_DIST["City"] == city) & (BUCKET_DIST["month"] == month)
    ][["AQI_Bucket", "pct"]].to_dict("records")

    return jsonify({
        "city":        city,
        "date":        str(today),
        "month_name":  MONTH_NAMES[month - 1],
        "predictions": predictions,
        "live": {
            "waqi":        waqi_data,
            "weather":     weather_data,
            "waqi_err":    waqi_err,
            "weather_err": weather_err,
        },
        "historical":    historical,
        "bucket_dist":   bd,
    })


@app.route("/api/historical")
def historical():
    city  = request.args.get("city",  "").strip()
    month = request.args.get("month", type=int)
    day   = request.args.get("day",   type=int)

    if not city:
        return jsonify({"error": "city is required"}), 400

    result = {}

    if month and day:
        # Try exact date first, fall back to month
        row = DAILY_STATS[
            (DAILY_STATS["City"] == city) &
            (DAILY_STATS["month"] == month) &
            (DAILY_STATS["day"] == day)
        ]
        result["level"] = "day" if not row.empty else "month"
        if row.empty:
            row = MONTHLY_STATS[
                (MONTHLY_STATS["City"] == city) &
                (MONTHLY_STATS["month"] == month)
            ]
    elif month:
        row = MONTHLY_STATS[
            (MONTHLY_STATS["City"] == city) & (MONTHLY_STATS["month"] == month)
        ]
        result["level"] = "month"
    else:
        return jsonify({"error": "month is required"}), 400

    if not row.empty:
        r = row.iloc[0]
        mean_aqi        = round(float(r["mean"]), 1)
        bucket, color   = aqi_bucket(mean_aqi)
        result.update({
            "mean":   mean_aqi,
            "min":    round(float(r["min"]),    1),
            "max":    round(float(r["max"]),    1),
            "median": round(float(r["median"]), 1),
            "count":  int(r["count"]),
            "bucket": bucket,
            "color":  color,
        })

    # Bucket distribution
    if month:
        bd = BUCKET_DIST[
            (BUCKET_DIST["City"] == city) & (BUCKET_DIST["month"] == month)
        ][["AQI_Bucket", "pct"]].to_dict("records")
        result["bucket_dist"] = bd

    return jsonify(result)


@app.route("/api/heatmap")
def heatmap():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "city is required"}), 400

    city_monthly = MONTHLY_STATS[MONTHLY_STATS["City"] == city]
    months_data  = []

    for m in range(1, 13):
        row = city_monthly[city_monthly["month"] == m]
        entry = {"month": m, "name": MONTH_NAMES[m - 1][:3], "full_name": MONTH_NAMES[m - 1]}

        if not row.empty:
            r             = row.iloc[0]
            mean_aqi      = round(float(r["mean"]), 1)
            bucket, color = aqi_bucket(mean_aqi)
            bd = BUCKET_DIST[
                (BUCKET_DIST["City"] == city) & (BUCKET_DIST["month"] == m)
            ][["AQI_Bucket", "pct"]].to_dict("records")
            dominant  = max(bd, key=lambda x: x["pct"])["AQI_Bucket"] if bd else "N/A"
            hazard_pct = sum(x["pct"] for x in bd if x["AQI_Bucket"] in ["Very Poor", "Severe"])
            entry.update({
                "mean":        mean_aqi,
                "min":         round(float(r["min"]), 1),
                "max":         round(float(r["max"]), 1),
                "bucket":      bucket,
                "color":       color,
                "dominant":    dominant,
                "hazard_pct":  round(hazard_pct, 1),
                "bucket_dist": bd,
            })
        months_data.append(entry)

    return jsonify({"city": city, "months": months_data})


if __name__ == "__main__":
    app.run(debug=True, port=5000)