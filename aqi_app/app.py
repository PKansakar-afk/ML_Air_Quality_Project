"""
AQI Prediction Flask App — Hindcast + Time Series
Run: python app.py
"""
import os, pickle, warnings
import requests
import numpy as np
import pandas as pd
from datetime import date
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)

app = Flask(__name__)

@app.errorhandler(404)
def not_found(e):    return jsonify({"error": str(e)}), 404
@app.errorhandler(500)
def server_error(e): return jsonify({"error": str(e)}), 500
@app.errorhandler(Exception)
def unhandled(e):    return jsonify({"error": f"Server error: {str(e)}"}), 500

# ── Load models & data ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
CSV_PATH   = os.path.join(BASE_DIR, "city_day_with_weather_complete.csv")

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
RF_MODEL        = MODELS["Random Forest"]
FEATURE_COLUMNS = _pkl("feature_columns.pkl")
STATS           = _pkl("historical_stats.pkl", folder=DATA_DIR)
print("done")

MONTHLY_STATS      = STATS["monthly_stats"]
BUCKET_DIST        = STATS["bucket_dist"]
CITY_MONTH_MEDIANS = STATS["city_month_medians"]
CITIES             = STATS["cities"]

print("Loading dataset...", end=" ", flush=True)
DF_FULL = pd.read_csv(CSV_PATH, parse_dates=["Date"])
DF_FULL = DF_FULL[DF_FULL["City"].isin(CITIES)].copy()
print(f"done  ({len(DF_FULL):,} rows)")

WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "demo")

CITY_COORDS = {
    "Ahmedabad": (23.0225,72.5714), "Aizawl": (23.7271,92.7176),
    "Amaravati": (16.5131,80.5165), "Amritsar": (31.6340,74.8723),
    "Bengaluru": (12.9716,77.5946), "Bhopal": (23.2599,77.4126),
    "Brajrajnagar": (21.8216,83.9216), "Chandigarh": (30.7333,76.7794),
    "Chennai": (13.0827,80.2707), "Coimbatore": (11.0168,76.9558),
    "Delhi": (28.6139,77.2090), "Ernakulam": (9.9816,76.2999),
    "Gurugram": (28.4601,77.0199), "Guwahati": (26.1445,91.7362),
    "Hyderabad": (17.3850,78.4867), "Jaipur": (26.9124,75.7873),
    "Jorapokhar": (23.7004,86.4125), "Kochi": (9.9312,76.2673),
    "Kolkata": (22.5726,88.3639), "Mumbai": (19.0760,72.8777),
    "Patna": (25.5941,85.1376), "Shillong": (25.5788,91.8933),
    "Talcher": (20.9509,85.2163), "Thiruvananthapuram": (8.5241,76.9366),
    "Visakhapatnam": (17.6868,83.2185),
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
POLL_COLS   = ["PM2.5","PM10","NO","NO2","NOx","CO","SO2","AQI"]
_ts_cache   = {}

# ── Helpers ────────────────────────────────────────────────────────────
def aqi_bucket(aqi):
    if   aqi <= 50:  return ("Good",        "#00c853")
    elif aqi <= 100: return ("Satisfactory", "#aeea00")
    elif aqi <= 200: return ("Moderate",     "#ffd600")
    elif aqi <= 300: return ("Poor",         "#ff6d00")
    elif aqi <= 400: return ("Very Poor",    "#dd2c00")
    else:            return ("Severe",       "#6a0080")

def fetch_weather(city):
    coords = CITY_COORDS.get(city)
    if not coords: return None, "No coords"
    lat, lon = coords
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto")
    try:
        c = requests.get(url, timeout=10).json().get("current", {})
        return {"temp": c.get("temperature_2m"), "humidity": c.get("relative_humidity_2m"),
                "wind": c.get("wind_speed_10m")}, None
    except Exception as e:
        return None, str(e)

def fetch_historical_air_quality(city, date_str):
    coords = CITY_COORDS.get(city)
    if not coords: return None, "No coords"
    lat, lon = coords
    url = (f"https://air-quality-api.open-meteo.com/v1/air-quality"
           f"?latitude={lat}&longitude={lon}"
           f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
           f"&start_date={date_str}&end_date={date_str}&timezone=auto")
    try:
        hourly = requests.get(url, timeout=15).json().get("hourly", {})
        if not hourly or not hourly.get("pm2_5"):
            return None, "No data for this date"
        def avg(vals):
            v = [x for x in (vals or []) if x is not None]
            return round(sum(v)/len(v), 3) if v else None
        co = avg(hourly.get("carbon_monoxide"))
        return {
            "pm25": avg(hourly.get("pm2_5")), "pm10": avg(hourly.get("pm10")),
            "no2":  avg(hourly.get("nitrogen_dioxide")),
            "so2":  avg(hourly.get("sulphur_dioxide")),
            "co":   round(co/1000, 4) if co else None,
            "no": None, "nox": None, "source": "Open-Meteo Archive"
        }, None
    except Exception as e:
        return None, str(e)

def get_medians(city, month):
    df  = CITY_MONTH_MEDIANS
    row = df[(df["City"]==city) & (df["month"]==month)]
    if row.empty: row = df[df["City"]==city]
    return row.iloc[0].to_dict() if not row.empty else {}

def build_feature_vector(city, target_date, waqi, weather):
    month = target_date.month
    med   = get_medians(city, month)
    def g(lk, mk, d=0.0):
        v = (waqi or {}).get(lk)
        if v is not None: return float(v)
        v = med.get(mk)
        return float(v) if v is not None else d

    pm25=g("pm25","PM2.5",50); pm10=g("pm10","PM10",80)
    no2=g("no2","NO2",30); so2=g("so2","SO2",15); co=g("co","CO",1.5)
    no=float(med.get("NO",20)); nox=float(med.get("NOx",45))
    aqi_e=g("aqi","AQI",150)
    temp=float((weather or {}).get("temp", med.get("Temp_Mean",25)))
    hum =float((weather or {}).get("humidity", med.get("Humidity_Mean",60)))
    wind=float((weather or {}).get("wind", med.get("Wind_Speed_Max",10)))
    ms=np.sin(2*np.pi*month/12); mc=np.cos(2*np.pi*month/12)
    live = (waqi or {}).get("aqi") is not None
    def lv(v, mk, d): return v if live else float(med.get(mk, d))
    al=lv(aqi_e,"AQI",150); p25l=lv(pm25,"PM2.5",50); p10l=lv(pm10,"PM10",80)
    n2l=lv(no2,"NO2",30); s2l=lv(so2,"SO2",15); col=lv(co,"CO",1.5)
    row = {
        "City":city, "PM2.5":pm25,"PM10":pm10,"NO":no,"NO2":no2,"NOx":nox,"CO":co,"SO2":so2,
        "Temp_Mean":temp,"Humidity_Mean":hum,"Wind_Speed_Max":wind,
        "year":target_date.year,"month":month,"day":target_date.day,
        "day_of_week":target_date.weekday(),"month_sin":ms,"month_cos":mc,
        **{f"PM2.5_lag{n}":p25l for n in [1,3,7]},
        **{f"PM10_lag{n}": p10l for n in [1,3,7]},
        **{f"NO_lag{n}":   float(med.get("NO",20)) for n in [1,3,7]},
        **{f"NO2_lag{n}":  n2l  for n in [1,3,7]},
        **{f"NOx_lag{n}":  float(med.get("NOx",45)) for n in [1,3,7]},
        **{f"CO_lag{n}":   col  for n in [1,3,7]},
        **{f"SO2_lag{n}":  s2l  for n in [1,3,7]},
        **{f"AQI_lag{n}":  al   for n in [1,3,7]},
        "PM2.5_roll7_mean":p25l,"PM2.5_roll30_mean":p25l,"PM2.5_roll7_std":5,
        "PM10_roll7_mean": p10l,"PM10_roll30_mean": p10l,"PM10_roll7_std":10,
        "NO_roll7_mean":float(med.get("NO",20)),"NO_roll30_mean":float(med.get("NO",20)),"NO_roll7_std":5,
        "NO2_roll7_mean":n2l,"NO2_roll30_mean":n2l,"NO2_roll7_std":5,
        "NOx_roll7_mean":float(med.get("NOx",45)),"NOx_roll30_mean":float(med.get("NOx",45)),"NOx_roll7_std":5,
        "CO_roll7_mean":col,"CO_roll30_mean":col,"CO_roll7_std":0.5,
        "SO2_roll7_mean":s2l,"SO2_roll30_mean":s2l,"SO2_roll7_std":3,
        "AQI_roll7_mean":al,"AQI_roll30_mean":al,"AQI_roll7_std":20,
    }
    df = pd.DataFrame([row])
    for c in FEATURE_COLUMNS:
        if c not in df.columns: df[c] = 0.0
    return df[FEATURE_COLUMNS]

def engineer_features(df_city):
    df = df_city.copy()
    poll = [c for c in POLL_COLS if c in df.columns]
    for col in poll:
        df[f"{col}_lag1"]        = df[col].shift(1)
        df[f"{col}_lag3"]        = df[col].shift(3)
        df[f"{col}_lag7"]        = df[col].shift(7)
        df[f"{col}_roll7_mean"]  = df[col].shift(1).rolling(7,  min_periods=1).mean()
        df[f"{col}_roll30_mean"] = df[col].shift(1).rolling(30, min_periods=1).mean()
        df[f"{col}_roll7_std"]   = df[col].shift(1).rolling(7,  min_periods=1).std().fillna(0)
    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["day"]         = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month_sin"]   = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]   = np.cos(2*np.pi*df["month"]/12)
    return df

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", cities=CITIES)


@app.route("/api/hindcast", methods=["POST"])
def hindcast():
    try:
        body     = request.get_json(force=True)
        city     = body.get("city","").strip()
        date_str = body.get("date","")
        if not city: return jsonify({"error":"city is required"}), 400
        try:    target_date = pd.to_datetime(date_str).date()
        except: return jsonify({"error":f"Invalid date: '{date_str}'"}), 400
        from datetime import date as dt
        if target_date > dt.today():
            return jsonify({"error":"Cannot hindcast a future date."}), 400

        TRAINING_END = pd.to_datetime("2020-06-30").date()
        month = target_date.month

        if target_date <= TRAINING_END:
            df_city = DF_FULL[DF_FULL["City"]==city].sort_values("Date").reset_index(drop=True)
            row     = df_city[df_city["Date"].dt.date==target_date]
            if row.empty: return jsonify({"error":f"No data for {city} on {date_str}"}), 404
            idx    = df_city[df_city["Date"].dt.date==target_date].index[0]
            window = engineer_features(df_city.iloc[max(0,idx-30):idx+1])
            tr     = window.iloc[[-1]].copy()
            tr["City"] = city
            for c in FEATURE_COLUMNS:
                if c not in tr.columns: tr[c] = 0.0
            X = tr[FEATURE_COLUMNS]
            r = row.iloc[0]
            actual_aqi = r.get("AQI")
            actual_readings = {
                "pm25":     round(float(r["PM2.5"]),2)  if pd.notna(r.get("PM2.5"))        else None,
                "pm10":     round(float(r["PM10"]),2)   if pd.notna(r.get("PM10"))         else None,
                "no2":      round(float(r["NO2"]),2)    if pd.notna(r.get("NO2"))          else None,
                "so2":      round(float(r["SO2"]),2)    if pd.notna(r.get("SO2"))          else None,
                "co":       round(float(r["CO"]),3)     if pd.notna(r.get("CO"))           else None,
                "temp":     round(float(r["Temp_Mean"]),1)    if pd.notna(r.get("Temp_Mean"))    else None,
                "humidity": round(float(r["Humidity_Mean"]),1) if pd.notna(r.get("Humidity_Mean"))else None,
                "wind":     round(float(r["Wind_Speed_Max"]),1)if pd.notna(r.get("Wind_Speed_Max"))else None,
            }
            air_source = "CPCB Dataset"
        else:
            air_data, air_err = fetch_historical_air_quality(city, date_str)
            if not air_data: return jsonify({"error":f"No pollutant data: {air_err}"}), 404
            weather_data, _  = fetch_weather(city)
            actual_aqi       = None
            X                = build_feature_vector(city, target_date, air_data, weather_data or {})
            actual_readings  = {
                "pm25":air_data.get("pm25"),"pm10":air_data.get("pm10"),
                "no2":air_data.get("no2"),  "so2":air_data.get("so2"),
                "co":air_data.get("co"),
                "temp":(weather_data or {}).get("temp"),
                "humidity":(weather_data or {}).get("humidity"),
                "wind":(weather_data or {}).get("wind"),
            }
            air_source = air_data.get("source","External API")

        predictions = {}
        for name, pipe in MODELS.items():
            try:
                val           = max(0.0, float(pipe.predict(X)[0]))
                bucket, color = aqi_bucket(val)
                predictions[name] = {"aqi":round(val,1),"bucket":bucket,"color":color}
            except Exception as e:
                predictions[name] = {"aqi":None,"error":str(e)}

        actual_bucket, actual_color = aqi_bucket(actual_aqi) if actual_aqi else ("Unknown","#888")

        hist_row = MONTHLY_STATS[(MONTHLY_STATS["City"]==city)&(MONTHLY_STATS["month"]==month)]
        historical = {}
        if not hist_row.empty:
            hr = hist_row.iloc[0]
            historical = {"mean":round(float(hr["mean"]),1),"min":round(float(hr["min"]),1),"max":round(float(hr["max"]),1)}

        bd = BUCKET_DIST[(BUCKET_DIST["City"]==city)&(BUCKET_DIST["month"]==month)][["AQI_Bucket","pct"]].to_dict("records")

        return jsonify({
            "city":city,"date":str(target_date),"month_name":MONTH_NAMES[month-1],
            "air_source":air_source,"predictions":predictions,
            "actual":{"aqi":round(float(actual_aqi),1) if actual_aqi else None,
                      "bucket":actual_bucket,"color":actual_color,"readings":actual_readings},
            "historical":historical,"bucket_dist":bd,
        })
    except Exception as e:
        return jsonify({"error":f"Hindcast failed: {str(e)}"}), 500


@app.route("/api/timeseries")
def timeseries():
    try:
        city = request.args.get("city","").strip()
        year = request.args.get("year", type=int)
        if not city: return jsonify({"error":"city is required"}), 400
        if not year or year < 2015 or year > 2020:
            return jsonify({"error":"year must be 2015–2020"}), 400

        cache_key = f"{city}_{year}"
        if cache_key in _ts_cache:
            return jsonify(_ts_cache[cache_key])

        df_city = (DF_FULL[DF_FULL["City"]==city]
                   .sort_values("Date").reset_index(drop=True)
                   .dropna(subset=["AQI"]))
        if df_city.empty: return jsonify({"error":f"No data for {city}"}), 404

        df_eng  = engineer_features(df_city).dropna(subset=["AQI_lag7"])
        df_year = df_eng[df_eng["Date"].dt.year==year].copy()
        if df_year.empty: return jsonify({"error":f"No data for {city} in {year}"}), 404

        df_year["City"] = city
        for col in FEATURE_COLUMNS:
            if col not in df_year.columns: df_year[col] = 0.0

        preds     = RF_MODEL.predict(df_year[FEATURE_COLUMNS])
        dates     = df_year["Date"].dt.strftime("%Y-%m-%d").tolist()
        actual    = [round(float(v),1) if pd.notna(v) else None for v in df_year["AQI"]]
        predicted = [round(max(0.0,float(p)),1) for p in preds]

        df_year["pred"] = predicted
        monthly = []
        for m in range(1,13):
            mdf = df_year[df_year["Date"].dt.month==m]
            if mdf.empty:
                monthly.append({"month":m,"name":MONTH_NAMES[m-1][:3],
                                 "actual_mean":None,"pred_mean":None}); continue
            am = round(float(mdf["AQI"].mean()),1)
            pm = round(float(mdf["pred"].mean()),1)
            bk, col = aqi_bucket(am)
            monthly.append({"month":m,"name":MONTH_NAMES[m-1][:3],
                            "actual_mean":am,"pred_mean":pm,"bucket":bk,"color":col})

        valid = [v for v in actual if v is not None]
        pairs = [(a,p) for a,p in zip(actual,predicted) if a is not None]
        rmse  = round(float(np.sqrt(np.mean([(a-p)**2 for a,p in pairs]))),2)
        mae   = round(float(np.mean([abs(a-p) for a,p in pairs])),2)
        ss_res= sum((a-p)**2 for a,p in pairs)
        ss_tot= sum((a-np.mean(valid))**2 for a in valid)
        r2    = round(1-ss_res/ss_tot,4) if ss_tot>0 else None

        result = {
            "city":city,"year":year,"dates":dates,
            "actual":actual,"predicted":predicted,"monthly":monthly,
            "stats":{"rmse":rmse,"mae":mae,"r2":r2,"n_days":len(dates),
                     "act_mean":round(float(np.mean(valid)),1),
                     "act_min": round(float(np.min(valid)),1),
                     "act_max": round(float(np.max(valid)),1)},
        }
        _ts_cache[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":f"Time series failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)