"""
=============================================================
  AQI PREDICTION — FULL PIPELINE
  Load → Clean → Feature Engineering → Train → Compare
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA (only once)
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Loading Data")
print("=" * 60)

df = pd.read_csv('city_day_with_weather_complete.csv')
print(f"  Raw shape: {df.shape}")

# Drop rows where target is missing
df = df.dropna(subset=['AQI'])

# Drop unnecessary columns
cols_to_drop = [c for c in ['AQI_Bucket', 'NH3', 'O3', 'Benzene',
                             'Toluene', 'Xylene'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

# Remove Lucknow (too many missing values)
df = df[df['City'] != 'Lucknow']

df['Date'] = pd.to_datetime(df['Date'])
print(f"  Clean shape: {df.shape}")
print(f"  Cities: {sorted(df['City'].unique())}")
print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ══════════════════════════════════════════════════════════
# STEP 2 — FILL NULLS WITH CITY-LEVEL MEDIAN
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Filling Null Values")
print("=" * 60)

cols_to_fill = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2']
cols_to_fill = [c for c in cols_to_fill if c in df.columns]

print("  Nulls before fill:")
print(df[cols_to_fill].isnull().sum().to_string())

for col in cols_to_fill:
    # City-level median first
    df[col] = df.groupby('City')[col].transform(
        lambda x: x.fillna(x.median())
    )
    # Global median fallback
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\n  Nulls after fill:")
print(df[cols_to_fill].isnull().sum().to_string())

# ══════════════════════════════════════════════════════════
# STEP 3 — SORT (must happen before lag features)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Sorting by City & Date")
print("=" * 60)

df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)
print("  Done.")

# ══════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Feature Engineering")
print("=" * 60)

# ── Lag & rolling features ─────────────────────────────────
poll_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'AQI']
poll_cols = [c for c in poll_cols if c in df.columns]

for col in poll_cols:
    grp = df.groupby('City')[col]

    # Lag features
    df[f'{col}_lag1'] = grp.shift(1)
    df[f'{col}_lag3'] = grp.shift(3)
    df[f'{col}_lag7'] = grp.shift(7)

    # Rolling mean
    df[f'{col}_roll7_mean']  = grp.shift(1).transform(
        lambda x: x.rolling(7,  min_periods=1).mean())
    df[f'{col}_roll30_mean'] = grp.shift(1).transform(
        lambda x: x.rolling(30, min_periods=1).mean())

    # Rolling std (volatility)
    df[f'{col}_roll7_std'] = grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).std())

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Shape after lag features: {df.shape}")

# ── Date parts ─────────────────────────────────────────────
df['year']       = df['Date'].dt.year
df['month']      = df['Date'].dt.month
df['day']        = df['Date'].dt.day
df['day_of_week']= df['Date'].dt.dayofweek

# ── Cyclical encoding for month ───────────────────────────
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

# ══════════════════════════════════════════════════════════
# STEP 5 — DEFINE X AND y
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Defining Features & Target")
print("=" * 60)

y = df['AQI']
X = df.drop(columns=['AQI', 'Date']).copy()

print(f"  Total features : {X.shape[1]}")
print(f"  Total samples  : {X.shape[0]}")
print("\n  Feature list:")
for i, col in enumerate(X.columns, 1):
    print(f"    {i:3}. {col}")

# ══════════════════════════════════════════════════════════
# STEP 6 — TRAIN / TEST SPLIT (time-based)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Train/Test Split (80/20 time-based)")
print("=" * 60)

split_index = int(0.8 * len(X))
X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(f"  Train size : {len(X_train):,}")
print(f"  Test size  : {len(X_test):,}")

# ══════════════════════════════════════════════════════════
# STEP 7 — PREPROCESSOR
# ══════════════════════════════════════════════════════════
categorical_features = ['City']
numeric_features     = [c for c in X.columns if c != 'City']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                      numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ══════════════════════════════════════════════════════════
# STEP 8 — TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Training Models")
print("=" * 60)

ridge_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1,
                           random_state=42, n_jobs=-1))
])

models = {
    "Ridge Regression": ridge_model,
    "Random Forest":    rf_model,
    "XGBoost":          xgb_model,
}

results  = []
all_preds = {}

for name, model in models.items():
    print(f"  Training {name} ...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_preds[name] = y_pred

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100

    results.append([name, round(mse,4), round(rmse,4),
                    round(mae,4), round(r2,6), round(mape,4)])
    print("done")

# ══════════════════════════════════════════════════════════
# STEP 8.5 — K-FOLD CROSS VALIDATION (evaluate all models)
# ══════════════════════════════════════════════════════════
from sklearn.model_selection import KFold, cross_validate

print("\n" + "=" * 60)
print("STEP 8.5 — K-Fold Cross Validation (k=5)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=False)  # shuffle=False keeps time order

cv_results = []

for name, model in models.items():
    print(f"  Running K-Fold on {name} ...", end=" ", flush=True)

    scores = cross_validate(
        model, X_train, y_train,
        cv=kf,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'mae':  'neg_mean_absolute_error',
            'r2':   'r2'
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model":       name,
        "CV_RMSE_mean": round(-scores['test_rmse'].mean(), 4),
        "CV_RMSE_std":  round( scores['test_rmse'].std(),  4),
        "CV_MAE_mean":  round(-scores['test_mae'].mean(),  4),
        "CV_R2_mean":   round( scores['test_r2'].mean(),   6),
    })
    print("done")

cv_df = pd.DataFrame(cv_results).sort_values("CV_RMSE_mean").reset_index(drop=True)

print("\n  K-Fold CV Results:")
print(cv_df.to_string(index=False))
cv_df.to_csv("kfold_cv_results.csv", index=False)
print("\n  Saved: kfold_cv_results.csv")

# ══════════════════════════════════════════════════════════
# STEP 9 — COMPARE ACCURACY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9 — Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(
    results, columns=["Model", "MSE", "RMSE", "MAE", "R2", "MAPE%"]
).sort_values("RMSE").reset_index(drop=True)

print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)
print("\n  Saved: model_comparison.csv")

best_model_name = results_df.iloc[0]["Model"]
print(f"\n  🏆 Best model: {best_model_name}")

# ══════════════════════════════════════════════════════════
# STEP 10 — ACCURACY PLOTS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10 — Plotting Accuracy")
print("=" * 60)

MODEL_COLORS = {
    "Ridge Regression": "#AB47BC",
    "Random Forest":    "#FF9800",
    "XGBoost":          "#2196F3",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Accuracy Comparison", fontsize=14, fontweight="bold")

model_names = results_df["Model"].tolist()
colors      = [MODEL_COLORS[m] for m in model_names]

# RMSE
ax = axes[0, 0]
bars = ax.bar(model_names, results_df["RMSE"], color=colors, alpha=0.85)
ax.set_title("RMSE (lower is better)", fontweight="bold")
for bar, val in zip(bars, results_df["RMSE"]):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3, f"{val:.2f}", ha="center", fontsize=9)
bars[0].set_edgecolor("gold"); bars[0].set_linewidth(2.5)
ax.grid(axis="y", alpha=0.3)

# R²
ax = axes[0, 1]
bars = ax.bar(model_names, results_df["R2"], color=colors, alpha=0.85)
ax.set_title("R² Score (higher is better)", fontweight="bold")
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, results_df["R2"]):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.005, f"{val:.4f}", ha="center", fontsize=9)
bars[0].set_edgecolor("gold"); bars[0].set_linewidth(2.5)
ax.grid(axis="y", alpha=0.3)

# MAE
ax = axes[1, 0]
bars = ax.bar(model_names, results_df["MAE"], color=colors, alpha=0.85)
ax.set_title("MAE (lower is better)", fontweight="bold")
for bar, val in zip(bars, results_df["MAE"]):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2, f"{val:.2f}", ha="center", fontsize=9)
bars[0].set_edgecolor("gold"); bars[0].set_linewidth(2.5)
ax.grid(axis="y", alpha=0.3)

# Actual vs Predicted (best model)
ax = axes[1, 1]
bp = all_preds[best_model_name]
ax.scatter(y_test, bp, alpha=0.3, s=6, color=MODEL_COLORS[best_model_name])
lims = [min(y_test.min(), bp.min()), max(y_test.max(), bp.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
ax.set_title(f"Actual vs Predicted — {best_model_name}", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: model_comparison_plot.png")

# ══════════════════════════════════════════════════════════
# STEP 11 — GRIDSEARCHCV ON BEST MODEL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 11 — GridSearchCV on {best_model_name}")
print("=" * 60)

if best_model_name == "Random Forest":
    param_grid = {
        'model__n_estimators':      [300, 500],
        'model__max_depth':         [15, 20, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf':  [1, 2],
        'model__max_features':      [0.6, 0.8, 'sqrt'],
    }
    best_pipeline = rf_model

elif best_model_name == "XGBoost":
    param_grid = {
        'model__n_estimators':     [300, 500],
        'model__max_depth':        [5, 7, 9],
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__subsample':        [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    }
    best_pipeline = xgb_model

else:  # Ridge
    param_grid   = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
    best_pipeline = ridge_model

grid_search = GridSearchCV(
    best_pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"  Running GridSearchCV (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

print(f"\n  Best params: {grid_search.best_params_}")

y_pred_tuned = grid_search.predict(X_test)
tuned_mse = mean_squared_error(y_test, y_pred_tuned)
tuned_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
tuned_r2     = r2_score(y_test, y_pred_tuned)
tuned_mae    = mean_absolute_error(y_test, y_pred_tuned)

print(f"  MSE  (after tuning) : {tuned_mse:.4f}")
print(f"\n  Before tuning → RMSE: {results_df.iloc[0]['RMSE']:.4f}  R²: {results_df.iloc[0]['R2']:.6f}")
print(f"  After  tuning → RMSE: {tuned_rmse:.4f}  R²: {tuned_r2:.6f}")
print(f"  MAE after tuning: {tuned_mae:.4f}")

# ══════════════════════════════════════════════════════════
# STEP 12 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Best model          : {best_model_name}")
print(f"  RMSE (after tuning) : {tuned_rmse:.4f}")
print(f"  MAE  (after tuning) : {tuned_mae:.4f}")
print(f"  R²   (after tuning) : {tuned_r2:.6f}")
print("\n  Output files:")
print("    • model_comparison.csv")
print("    • model_comparison_plot.png")
print("=" * 60)
