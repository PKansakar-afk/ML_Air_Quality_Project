"""
AQI Prediction Flask App
Run: python app.py
Set env var WAQI_TOKEN before running (free at aqicn.org/data-platform/token/)
"""
from dotenv import load_dotenv
import os
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import date
from flask import Flask, request, jsonify, render_template

load_dotenv()

app = Flask(__name__)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def unhandled(e):
    return jsonify({"error": f"Server error: {str(e)}"}), 500

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

print("WAQI TOKEN:", WAQI_TOKEN)

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
    coords = CITY_COORDS.get(city)
    if not coords:
        return None, f"No coordinates for {city}"
    lat, lon = coords
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"

    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        if j.get("status") == "ok":
            d    = j["data"]
            iaqi = d.get("iaqi", {})

            raw_pm25 = iaqi.get("pm25", {}).get("v")
            raw_pm10 = iaqi.get("pm10", {}).get("v")
            raw_no2  = iaqi.get("no2",  {}).get("v")
            raw_so2  = iaqi.get("so2",  {}).get("v")
            raw_co   = iaqi.get("co",   {}).get("v")

            return {
                "pm25": round(raw_pm25, 2)         if raw_pm25 is not None else None,
                "pm10": round(raw_pm10, 2)         if raw_pm10 is not None else None,
                "no2":  round(raw_no2  * 1.88, 2)  if raw_no2  is not None else None,
                "so2":  round(raw_so2  * 2.62, 2)  if raw_so2  is not None else None,
                "co":   round(raw_co   * 1.145, 3) if raw_co   is not None else None,
                "no":   None,
                "nox":  None,
                "aqi":     d.get("aqi"),
                "station": d["city"]["name"],
                "updated": d["time"]["s"],
            }, None
        return None, j.get("data", "WAQI returned non-ok status")
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
    month = target_date.month
    med   = get_medians(city, month)

    def g(live_key, med_key, default=0.0):
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

    # ── Key fix: use live values for lag/rolling when available ───────
    # Live reading is a much better proxy for recent days than a
    # 5-year historical monthly average.
    have_live = (waqi or {}).get("aqi") is not None

    def lag_val(live_val, med_key, med_fallback):
        """Use live reading if available, else historical median."""
        return live_val if have_live else float(med.get(med_key, med_fallback))

    aqi_lag  = lag_val(aqi_est, "AQI",   150.0)
    pm25_lag = lag_val(pm25,    "PM2.5",  50.0)
    pm10_lag = lag_val(pm10,    "PM10",   80.0)
    no2_lag  = lag_val(no2,     "NO2",    30.0)
    so2_lag  = lag_val(so2,     "SO2",    15.0)
    co_lag   = lag_val(co,      "CO",      1.5)
    no_lag   = float(med.get("NO",  20.0))   # never live — not in WAQI
    nox_lag  = float(med.get("NOx", 45.0))   # never live — not in WAQI

    row = {
        "City": city,
        # ── Base pollutants ──────────────────────────────────────────
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
        "NOx": nox, "CO": co, "SO2": so2,
        # ── Weather ──────────────────────────────────────────────────
        "Temp_Mean": temp, "Humidity_Mean": humidity, "Wind_Speed_Max": wind,
        # ── Date ─────────────────────────────────────────────────────
        "year": target_date.year, "month": month,
        "day": target_date.day, "day_of_week": target_date.weekday(),
        "month_sin": ms, "month_cos": mc,
        # ── Lag features (live reading used when available) ──────────
        "PM2.5_lag1": pm25_lag, "PM2.5_lag3": pm25_lag, "PM2.5_lag7": pm25_lag,
        "PM10_lag1":  pm10_lag, "PM10_lag3":  pm10_lag, "PM10_lag7":  pm10_lag,
        "NO_lag1":    no_lag,   "NO_lag3":    no_lag,   "NO_lag7":    no_lag,
        "NO2_lag1":   no2_lag,  "NO2_lag3":   no2_lag,  "NO2_lag7":   no2_lag,
        "NOx_lag1":   nox_lag,  "NOx_lag3":   nox_lag,  "NOx_lag7":   nox_lag,
        "CO_lag1":    co_lag,   "CO_lag3":    co_lag,   "CO_lag7":    co_lag,
        "SO2_lag1":   so2_lag,  "SO2_lag3":   so2_lag,  "SO2_lag7":   so2_lag,
        "AQI_lag1":   aqi_lag,  "AQI_lag3":   aqi_lag,  "AQI_lag7":   aqi_lag,
        # ── Rolling means (live reading when available, else median) ──
        "PM2.5_roll7_mean":  pm25_lag, "PM2.5_roll30_mean": pm25_lag, "PM2.5_roll7_std":  5.0,
        "PM10_roll7_mean":   pm10_lag, "PM10_roll30_mean":  pm10_lag, "PM10_roll7_std":   10.0,
        "NO_roll7_mean":     no_lag,   "NO_roll30_mean":    no_lag,   "NO_roll7_std":     5.0,
        "NO2_roll7_mean":    no2_lag,  "NO2_roll30_mean":   no2_lag,  "NO2_roll7_std":    5.0,
        "NOx_roll7_mean":    nox_lag,  "NOx_roll30_mean":   nox_lag,  "NOx_roll7_std":    5.0,
        "CO_roll7_mean":     co_lag,   "CO_roll30_mean":    co_lag,   "CO_roll7_std":     0.5,
        "SO2_roll7_mean":    so2_lag,  "SO2_roll30_mean":   so2_lag,  "SO2_roll7_std":    3.0,
        "AQI_roll7_mean":    aqi_lag,  "AQI_roll30_mean":   aqi_lag,  "AQI_roll7_std":    20.0,
    }

    df = pd.DataFrame([row])
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]

def fetch_historical_air_quality(city: str, date_str: str) -> tuple:
    """
    Fetch historical pollutant data from Open-Meteo Air Quality Archive.
    Covers ~2022 onwards well; 2021 is partial depending on city.
    Returns daily averages matching CPCB units.
    """
    coords = CITY_COORDS.get(city)
    if not coords:
        return None, "No coordinates"
    lat, lon = coords

    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
        f"&start_date={date_str}&end_date={date_str}"
        f"&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=15)
        j = r.json()
        hourly = j.get("hourly", {})
        if not hourly or not hourly.get("pm2_5"):
            return None, "No air quality data for this date/location"

        def daily_avg(values):
            valid = [v for v in (values or []) if v is not None]
            return round(sum(valid) / len(valid), 3) if valid else None

        pm25 = daily_avg(hourly.get("pm2_5"))
        pm10 = daily_avg(hourly.get("pm10"))
        no2  = daily_avg(hourly.get("nitrogen_dioxide"))   # μg/m³ ✓
        so2  = daily_avg(hourly.get("sulphur_dioxide"))    # μg/m³ ✓
        co_ugm3 = daily_avg(hourly.get("carbon_monoxide")) # μg/m³ → need mg/m³
        co   = round(co_ugm3 / 1000, 4) if co_ugm3 else None  # ÷1000 → mg/m³

        return {
            "pm25": pm25, "pm10": pm10,
            "no2":  no2,  "so2":  so2,
            "co":   co,
            "no":   None, "nox":  None,  # not available
            "source": "Open-Meteo Air Quality Archive"
        }, None
    except Exception as e:
        return None, str(e)


def fetch_historical_openaq(city: str, date_str: str) -> tuple:
    """
    Fallback: OpenAQ API for 2021 and earlier dates.
    Free, no key needed for basic usage.
    """
    coords = CITY_COORDS.get(city)
    if not coords:
        return None, "No coordinates"

    # OpenAQ v3 — search by coordinates, date range
    date_from = f"{date_str}T00:00:00Z"
    date_to   = f"{date_str}T23:59:59Z"
    lat, lon  = coords

    url = (
        f"https://api.openaq.org/v3/measurements"
        f"?coordinates={lat},{lon}&radius=25000"
        f"&datetime_from={date_from}&datetime_to={date_to}"
        f"&limit=1000"
    )
    try:
        r = requests.get(url, timeout=15, headers={"Accept": "application/json"})
        results = r.json().get("results", [])
        if not results:
            return None, "No OpenAQ data for this date/location"

        from collections import defaultdict
        readings = defaultdict(list)
        for m in results:
            param = m.get("parameter", "").lower()
            value = m.get("value")
            unit  = m.get("unit", "")
            if value is not None and value >= 0:
                readings[param].append((value, unit))

        def avg(key):
            vals = [v for v, u in readings.get(key, [])]
            return round(sum(vals)/len(vals), 3) if vals else None

        # OpenAQ returns μg/m³ for most, but CO can be in ppm — check unit
        co_readings = readings.get("co", [])
        co_val = None
        if co_readings:
            val, unit = co_readings[0]
            avg_co = sum(v for v,u in co_readings) / len(co_readings)
            # Convert to mg/m³ to match CPCB
            if "ppm" in unit.lower():
                co_val = round(avg_co * 1.145, 4)  # ppm → mg/m³
            else:
                co_val = round(avg_co / 1000, 4)   # μg/m³ → mg/m³

        return {
            "pm25": avg("pm25"), "pm10": avg("pm10"),
            "no2":  avg("no2"),  "so2":  avg("so2"),
            "co":   co_val,
            "no":   avg("no"),   "nox":  None,
            "source": "OpenAQ"
        }, None
    except Exception as e:
        return None, str(e)


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

@app.route("/api/hindcast", methods=["POST"])
def hindcast():
    try:
        body     = request.get_json(force=True)
        city     = body.get("city", "").strip()
        date_str = body.get("date", "")

        if not city:
            return jsonify({"error": "city is required"}), 400
        try:
            target_date = pd.to_datetime(date_str).date()
        except Exception:
            return jsonify({"error": f"Invalid date: {date_str}"}), 400

        TRAINING_END = pd.to_datetime("2020-06-30").date()
        month        = target_date.month
        air_source   = None

        from datetime import date as date_type
        if target_date > date_type.today():
            return jsonify({"error": "Cannot hindcast a future date. Use Nowcast instead."}), 400

        if target_date <= TRAINING_END:
            # ── Use actual dataset rows (most accurate) ────────
            df_full = pd.read_csv(
                os.path.join(BASE_DIR, "city_day_with_weather_complete.csv"),
                parse_dates=["Date"]
            )
            df_city = df_full[df_full["City"] == city].sort_values("Date").reset_index(drop=True)
            row     = df_city[df_city["Date"].dt.date == target_date]

            if row.empty:
                return jsonify({"error": f"No dataset row for {city} on {date_str}"}), 404

            row_idx = df_city[df_city["Date"].dt.date == target_date].index[0]
            window  = df_city.iloc[max(0, row_idx-30): row_idx+1].copy()

            poll_cols = [c for c in ["PM2.5","PM10","NO","NO2","NOx","CO","SO2","AQI"] if c in window.columns]
            for col in poll_cols:
                window[f"{col}_lag1"]        = window[col].shift(1)
                window[f"{col}_lag3"]        = window[col].shift(3)
                window[f"{col}_lag7"]        = window[col].shift(7)
                window[f"{col}_roll7_mean"]  = window[col].shift(1).rolling(7,  min_periods=1).mean()
                window[f"{col}_roll30_mean"] = window[col].shift(1).rolling(30, min_periods=1).mean()
                window[f"{col}_roll7_std"]   = window[col].shift(1).rolling(7,  min_periods=1).std().fillna(0)

            window["year"]        = window["Date"].dt.year
            window["month"]       = window["Date"].dt.month
            window["day"]         = window["Date"].dt.day
            window["day_of_week"] = window["Date"].dt.dayofweek
            window["month_sin"]   = np.sin(2 * np.pi * window["month"] / 12)
            window["month_cos"]   = np.cos(2 * np.pi * window["month"] / 12)

            target_row = window.iloc[[-1]].copy()
            for col in FEATURE_COLUMNS:
                if col not in target_row.columns:
                    target_row[col] = 0.0
            X = target_row[FEATURE_COLUMNS]

            r          = row.iloc[0]
            actual_aqi = r.get("AQI")
            actual_readings = {
                "pm25":     round(float(r["PM2.5"]), 2)        if pd.notna(r.get("PM2.5"))        else None,
                "pm10":     round(float(r["PM10"]),  2)        if pd.notna(r.get("PM10"))         else None,
                "no2":      round(float(r["NO2"]),   2)        if pd.notna(r.get("NO2"))          else None,
                "so2":      round(float(r["SO2"]),   2)        if pd.notna(r.get("SO2"))          else None,
                "co":       round(float(r["CO"]),    3)        if pd.notna(r.get("CO"))           else None,
                "temp":     round(float(r["Temp_Mean"]),    1) if pd.notna(r.get("Temp_Mean"))    else None,
                "humidity": round(float(r["Humidity_Mean"]),1) if pd.notna(r.get("Humidity_Mean"))else None,
                "wind":     round(float(r["Wind_Speed_Max"]),1)if pd.notna(r.get("Wind_Speed_Max"))else None,
            }
            air_source = "CPCB Dataset (training data)"

        else:
            # ── Post-2020: fetch from APIs ────────────────────
            # Try Open-Meteo first, fall back to OpenAQ
            air_data, air_err = fetch_historical_air_quality(city, date_str)
            if not air_data:
                air_data, air_err = fetch_historical_openaq(city, date_str)
            if not air_data:
                return jsonify({"error": f"No historical pollutant data available: {air_err}"}), 404

            weather_data, _ = fetch_weather(city)   # current weather as proxy
            air_source      = air_data.get("source", "External API")
            actual_aqi      = None   # unknown for post-training dates

            X = build_feature_vector(city, target_date, air_data, weather_data or {})

            actual_readings = {
                "pm25":     air_data.get("pm25"),
                "pm10":     air_data.get("pm10"),
                "no2":      air_data.get("no2"),
                "so2":      air_data.get("so2"),
                "co":       air_data.get("co"),
                "temp":     (weather_data or {}).get("temp"),
                "humidity": (weather_data or {}).get("humidity"),
                "wind":     (weather_data or {}).get("wind"),
            }

        # ── Predict ────────────────────────────────────────────
        predictions = {}
        for name, pipe in MODELS.items():
            try:
                val           = max(0.0, float(pipe.predict(X)[0]))
                bucket, color = aqi_bucket(val)
                predictions[name] = {"aqi": round(val,1), "bucket": bucket, "color": color}
            except Exception as e:
                predictions[name] = {"aqi": None, "error": str(e)}

        actual_bucket, actual_color = (
            aqi_bucket(actual_aqi) if actual_aqi else ("Unknown", "#888")
        )

        hist_row = MONTHLY_STATS[
            (MONTHLY_STATS["City"] == city) & (MONTHLY_STATS["month"] == month)
        ]
        historical = {}
        if not hist_row.empty:
            hr = hist_row.iloc[0]
            historical = {"mean": round(float(hr["mean"]),1),
                        "min":  round(float(hr["min"]), 1),
                        "max":  round(float(hr["max"]), 1)}

        bd = BUCKET_DIST[
            (BUCKET_DIST["City"] == city) & (BUCKET_DIST["month"] == month)
        ][["AQI_Bucket","pct"]].to_dict("records")

        return jsonify({
            "city":        city,
            "date":        str(target_date),
            "month_name":  MONTH_NAMES[month - 1],
            "predictions": predictions,
            "air_source":  air_source,
            "actual": {
                "aqi":      round(float(actual_aqi),1) if actual_aqi else None,
                "bucket":   actual_bucket,
                "color":    actual_color,
                "readings": actual_readings,
            },
            "historical":  historical,
            "bucket_dist": bd,
        })
    except Exception as e:
        return jsonify({"error": f"Hindcast failed: {str(e)}"}), 500

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)