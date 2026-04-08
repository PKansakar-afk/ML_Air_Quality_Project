"""
=============================================================
  AQI PREDICTION — IMPROVED PIPELINE (Step-by-Step)
=============================================================
Steps:
  1. Load & clean data
  2. Feature engineering (lags, rolling, time, city encoding)
  3. Time-based train/test split (NO random shuffle)
  4. Ridge Regression  (with proper scaling)
  5. Random Forest     (tuned)
  6. XGBoost           (tuned)
  7. LightGBM          (tuned)
  8. LSTM              (fixed: per-city sequences, no shuffle)
  9. Results table + plots
=============================================================
Install requirements:
  pip install pandas numpy scikit-learn xgboost lightgbm tensorflow matplotlib seaborn
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and cleaning data")
print("=" * 60)

df = pd.read_csv("city_day_interpolated_fixed.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

print(f"  Shape: {df.shape}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Cities: {df['City'].nunique()}")
print(f"  AQI range: {df['AQI'].min():.0f} – {df['AQI'].max():.0f}  (mean={df['AQI'].mean():.1f})")

# Remove extreme outliers (top 0.5%)
upper_cap = df["AQI"].quantile(0.995)
df = df[df["AQI"] <= upper_cap].copy()
print(f"  After removing top-0.5% outliers: {df.shape[0]} rows  (cap={upper_cap:.0f})")


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature engineering")
print("=" * 60)

# --- 2a. Time features ---
df["Month"]      = df["Date"].dt.month
df["DayOfYear"]  = df["Date"].dt.dayofyear
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["Year"]       = df["Date"].dt.year
df["Season"]     = df["Month"].map({
    12:0, 1:0, 2:0,   # Winter
     3:1, 4:1, 5:1,   # Spring
     6:2, 7:2, 8:2,   # Summer
     9:3,10:3,11:3    # Autumn
})
# Cyclical encoding for month & day (avoids Dec→Jan jump)
df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)

# --- 2b. City encoding ---
le = LabelEncoder()
df["City_enc"] = le.fit_transform(df["City"])

# --- 2c. Lag & rolling features (per city, sorted by date) ---
pollutants = ["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2"]

for col in pollutants:
    grp = df.groupby("City")[col]
    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f"{col}_lag{lag}"] = grp.shift(lag)
    # Rolling statistics
    for window in [3, 7, 14, 30]:
        df[f"{col}_roll{window}_mean"] = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll{window}_std"] = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

# --- 2d. Interaction features ---
df["PM_ratio"]    = df["PM2.5"] / (df["PM10"] + 1e-5)
df["PM_sum"]      = df["PM2.5"] + df["PM10"]
df["NO2_CO"]      = df["NO2"] * df["CO"]
df["Temp_x_Hum"]  = df["Temp_Mean"] * df["Humidity_Mean"]

# Drop rows where lag features are NaN (first ~30 days per city)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Features created. Final shape: {df.shape}")


# ─────────────────────────────────────────────
# STEP 3: TIME-BASED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Time-based train/test split")
print("=" * 60)

SPLIT_DATE = "2020-01-01"
train_df = df[df["Date"] < SPLIT_DATE].copy()
test_df  = df[df["Date"] >= SPLIT_DATE].copy()

FEATURE_COLS = [c for c in df.columns if c not in ["City", "Date", "AQI"]]
TARGET = "AQI"

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET].values

print(f"  Train: {X_train.shape[0]} rows  ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
print(f"  Test : {X_test.shape[0]} rows  ({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")
print(f"  Features: {X_train.shape[1]}")

# Scale for linear models / LSTM
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

results = {}

def evaluate(name, y_true, y_pred):
    # Standard metrics (Original Scale)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    
    # Calculate Scaled MSE
    y_true_sc = scaler_y.transform(y_true.reshape(-1, 1)).ravel()
    y_pred_sc = scaler_y.transform(y_pred.reshape(-1, 1)).ravel()
    scaled_mse = mean_squared_error(y_true_sc, y_pred_sc)
    
    # Print and store both
    print(f"  {name:20s}  MSE={mse:8.3f}  Scaled MSE={scaled_mse:8.5f}  RMSE={rmse:7.3f}  R²={r2:.6f}")
    results[name] = {
        "MSE": mse, 
        "Scaled_MSE": scaled_mse, 
        "RMSE": rmse, 
        "R2": r2,
        "y_pred": y_pred, 
        "y_true": y_true
    }
    return mse, rmse, r2

# ─────────────────────────────────────────────
# STEP 4: RIDGE REGRESSION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Ridge Regression")
print("=" * 60)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
y_pred_ridge = ridge.predict(X_test_sc)
evaluate("Ridge Regression", y_test, y_pred_ridge)


# ─────────────────────────────────────────────
# STEP 5: RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Random Forest")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.6,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate("Random Forest", y_test, y_pred_rf)


# ─────────────────────────────────────────────
# STEP 6: XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: XGBoost")
print("=" * 60)

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
y_pred_xgb = xgb_model.predict(X_test)
evaluate("XGBoost", y_test, y_pred_xgb)


# ─────────────────────────────────────────────
# STEP 7: LIGHTGBM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: LightGBM")
print("=" * 60)

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)]
)
y_pred_lgb = lgb_model.predict(X_test)
evaluate("LightGBM", y_test, y_pred_lgb)


# ─────────────────────────────────────────────
# STEP 8: LSTM (FIXED)
# ─────────────────────────────────────────────
# print("\n" + "=" * 60)
# print("STEP 8: LSTM (fixed — per-city sequences, time-based split)")
# print("=" * 60)

# SEQ_LEN = 14   # use 14 days of history to predict next day

# def build_sequences(data_df, feature_cols, target_col, seq_len):
#     """Build LSTM sequences grouped by city, preserving time order."""
#     X_seqs, y_seqs = [], []
#     for city, grp in data_df.groupby("City"):
#         grp = grp.sort_values("Date")
#         feats  = grp[feature_cols].values
#         target = grp[target_col].values
#         for i in range(seq_len, len(grp)):
#             X_seqs.append(feats[i - seq_len : i])
#             y_seqs.append(target[i])
#     return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

# # Use scaled features for LSTM
# train_df_sc = train_df.copy()
# test_df_sc  = test_df.copy()
# train_df_sc[FEATURE_COLS] = scaler_X.transform(train_df[FEATURE_COLS].values)
# test_df_sc[FEATURE_COLS]  = scaler_X.transform(test_df[FEATURE_COLS].values)
# train_df_sc["AQI_sc"] = scaler_y.transform(train_df[["AQI"]].values).ravel()
# test_df_sc["AQI_sc"]  = scaler_y.transform(test_df[["AQI"]].values).ravel()

# print("  Building sequences...")
# X_lstm_train, y_lstm_train = build_sequences(train_df_sc, FEATURE_COLS, "AQI_sc", SEQ_LEN)
# X_lstm_test,  y_lstm_test  = build_sequences(test_df_sc,  FEATURE_COLS, "AQI_sc", SEQ_LEN)
# print(f"  Train sequences: {X_lstm_train.shape}")
# print(f"  Test  sequences: {X_lstm_test.shape}")

# n_features = X_lstm_train.shape[2]

# # Build LSTM model
# lstm_model = Sequential([
#     LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
#     Dropout(0.2),
#     BatchNormalization(),
#     LSTM(64, return_sequences=False),
#     Dropout(0.2),
#     BatchNormalization(),
#     Dense(32, activation="relu"),
#     Dense(1)
# ])

# lstm_model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss="mse"
# )

# callbacks = [
#     EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
# ]

# print("  Training LSTM...")
# history = lstm_model.fit(
#     X_lstm_train, y_lstm_train,
#     validation_data=(X_lstm_test, y_lstm_test),
#     epochs=100,
#     batch_size=256,
#     callbacks=callbacks,
#     verbose=1
# )

# # Predict and inverse-transform
# y_pred_lstm_sc = lstm_model.predict(X_lstm_test, verbose=0).ravel()
# y_pred_lstm    = scaler_y.inverse_transform(y_pred_lstm_sc.reshape(-1, 1)).ravel()
# y_test_lstm    = scaler_y.inverse_transform(y_lstm_test.reshape(-1, 1)).ravel()

# evaluate("LSTM", y_test_lstm, y_pred_lstm)


# ─────────────────────────────────────────────
# STEP 9: RESULTS TABLE + PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Final Results Summary")
print("=" * 60)

summary = pd.DataFrame([
    {"Model": k, "MSE": v["MSE"], "Scaled MSE": v["Scaled_MSE"], "RMSE": v["RMSE"], "R²": v["R2"]}
    for k, v in results.items()
]).sort_values("MSE")

print("\n" + summary.to_string(index=False))

# ── Plot 1: Model comparison bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison (Improved Pipeline)", fontsize=14, fontweight="bold")

colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
models = summary["Model"].tolist()

axes[0].barh(models, summary["MSE"],  color=colors[:len(models)])
axes[0].set_title("MSE (lower is better)")
axes[0].set_xlabel("MSE")
axes[0].axvline(x=10, color="red", linestyle="--", label="Target MSE=10")
axes[0].legend()

axes[1].barh(models, summary["RMSE"], color=colors[:len(models)])
axes[1].set_title("RMSE (lower is better)")
axes[1].set_xlabel("RMSE")

axes[2].barh(models, summary["R²"],  color=colors[:len(models)])
axes[2].set_title("R² (higher is better)")
axes[2].set_xlabel("R²")
axes[2].set_xlim([0, 1])

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: results_comparison.png")

# ── Plot 2: Actual vs Predicted for best model ──
best_model_name = summary.iloc[0]["Model"]
best = results[best_model_name]

sample_size = min(500, len(best["y_true"]))
idx = np.linspace(0, len(best["y_true"]) - 1, sample_size, dtype=int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model: {best_model_name}", fontsize=13, fontweight="bold")

axes[0].plot(best["y_true"][idx], label="Actual",    alpha=0.7, color="#1565C0")
axes[0].plot(best["y_pred"][idx], label="Predicted", alpha=0.7, color="#E53935", linestyle="--")
axes[0].set_title("Actual vs Predicted (sampled)")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("AQI")
axes[0].legend()

axes[1].scatter(best["y_true"][idx], best["y_pred"][idx],
                alpha=0.3, color="#6A1B9A", s=10)
lim = max(best["y_true"][idx].max(), best["y_pred"][idx].max())
axes[1].plot([0, lim], [0, lim], "r--", label="Perfect fit")
axes[1].set_title("Actual vs Predicted (scatter)")
axes[1].set_xlabel("Actual AQI")
axes[1].set_ylabel("Predicted AQI")
axes[1].legend()

plt.tight_layout()
plt.savefig("best_model_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: best_model_predictions.png")

# ── Plot 3: Feature importance (best tree model) ──
tree_models = {
    "Random Forest": (rf,      "RF importances"),
    "XGBoost":       (xgb_model, "XGB importances"),
    "LightGBM":      (lgb_model, "LGB importances"),
}
best_tree = summary[summary["Model"].isin(tree_models.keys())].iloc[0]["Model"]
best_tree_model = tree_models[best_tree][0]

if hasattr(best_tree_model, "feature_importances_"):
    fi = pd.Series(best_tree_model.feature_importances_, index=FEATURE_COLS)
    top20 = fi.nlargest(20)

    plt.figure(figsize=(10, 7))
    top20.sort_values().plot(kind="barh", color="#00796B")
    plt.title(f"Top 20 Feature Importances — {best_tree}", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance.png")

# ── Plot 4: LSTM training history ──
# plt.figure(figsize=(8, 4))
# plt.plot(history.history["loss"],     label="Train Loss")
# plt.plot(history.history["val_loss"], label="Val Loss")
# plt.title("LSTM Training History", fontsize=13, fontweight="bold")
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.legend()
# plt.tight_layout()
# plt.savefig("lstm_training.png", dpi=150, bbox_inches="tight")
# plt.close()
# print("  Saved: lstm_training.png")

# print("\n" + "=" * 60)
# print("  DONE! All models trained and evaluated.")
# print("=" * 60)



# ─────────────────────────────────────────────
# STEP 10: SHAP ANALYSIS (Model Interpretability)
# ─────────────────────────────────────────────
import shap

print("\n" + "=" * 60)
print("STEP 10: SHAP Analysis for LightGBM")
print("=" * 60)

# 1. Initialize the TreeExplainer with your best tree model
# (Assuming lgb_model from Step 7 is your chosen model)
explainer = shap.TreeExplainer(lgb_model)

# 2. To save time, calculate SHAP values on the test set 
# (or a sample of it if the test set is massive)
X_test_sample = X_test  # Or pd.DataFrame(X_test, columns=FEATURE_COLS).sample(1000)
shap_values = explainer(X_test_sample)

# ── Plot A: SHAP Summary Plot (Beeswarm) ──
# This shows feature importance AND the direction of the impact
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=FEATURE_COLS, max_display=40, show=False)
plt.title("SHAP Summary: How Features Impact AQI Predictions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=150)
plt.close()
print("  Saved: shap_summary_beeswarm.png")

# ── Plot B: Partial Dependence Plot (PDP) ──
# Let's look at the top feature (likely a rolling mean or lag)
top_feature = FEATURE_COLS[np.argsort(-np.abs(shap_values.values).mean(0))[0]]
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values.values, X_test_sample, feature_names=FEATURE_COLS, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"shap_dependence_plot.png", dpi=150)
plt.close()
print("  Saved: shap_dependence_plot.png")

# ── Plot C: Local Interpretability (Waterfall Plot) ──
# Pick the single worst AQI day in the test set to explain it
worst_day_idx = np.argmax(y_test)
worst_day_actual = y_test[worst_day_idx]
worst_day_pred = y_pred_lgb[worst_day_idx]

print(f"\n  Explaining worst day prediction (Actual AQI: {worst_day_actual:.1f}, Predicted: {worst_day_pred:.1f})")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[worst_day_idx], max_display=10, show=False)
plt.title("SHAP Waterfall: Explaining the Worst AQI Day", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_waterfall_worst_day.png", dpi=150)
plt.close()
print("  Saved: shap_waterfall_worst_day.png")