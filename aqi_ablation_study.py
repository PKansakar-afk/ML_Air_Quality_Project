"""
=============================================================
  AQI PREDICTION — ABLATION STUDY (ALL MODELS)
=============================================================
Systematically removes feature groups to understand their
contribution to model performance.

Models: Ridge, XGBoost, LightGBM, RandomForest

Feature Groups:
  1. Lag features (lag1, lag2, lag3, lag7, lag14)
  2. Rolling statistics (mean, std for 3, 7, 14, 30 days)
  3. Time features (Month, DayOfYear, Season, cyclical)
  4. City encoding
  5. Interaction features (PM_ratio, PM_sum, etc.)
  6. Base pollutants (current day values)
  7. Meteorological
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading and preparing data for ablation study")
print("=" * 60)

df = pd.read_csv("city_day_interpolated_fixed.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"]).reset_index(drop=True)

# Remove extreme outliers
upper_cap = df["AQI"].quantile(0.995)
df = df[df["AQI"] <= upper_cap].copy()

# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Time features
df["Month"]      = df["Date"].dt.month
df["DayOfYear"]  = df["Date"].dt.dayofyear
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["Year"]       = df["Date"].dt.year
df["Season"]     = df["Month"].map({
    12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3
})
df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)

# City encoding
le = LabelEncoder()
df["City_enc"] = le.fit_transform(df["City"])

# Lag & rolling features
pollutants = ["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2"]

for col in pollutants:
    grp = df.groupby("City")[col]
    for lag in [1, 2, 3, 7, 14]:
        df[f"{col}_lag{lag}"] = grp.shift(lag)
    for window in [3, 7, 14, 30]:
        df[f"{col}_roll{window}_mean"] = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll{window}_std"] = grp.shift(1).transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

# Interaction features
df["PM_ratio"]    = df["PM2.5"] / (df["PM10"] + 1e-5)
df["PM_sum"]      = df["PM2.5"] + df["PM10"]
df["NO2_CO"]      = df["NO2"] * df["CO"]
df["Temp_x_Hum"]  = df["Temp_Mean"] * df["Humidity_Mean"]

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────
# STEP 3: DEFINE FEATURE GROUPS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Defining feature groups for ablation")
print("=" * 60)

all_features = [c for c in df.columns if c not in ["City", "Date", "AQI"]]

LAG_FEATURES         = [c for c in all_features if "_lag" in c]
ROLLING_FEATURES     = [c for c in all_features if "_roll" in c]
TIME_FEATURES        = ["Month", "DayOfYear", "DayOfWeek", "Year", "Season",
                        "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"]
CITY_FEATURES        = ["City_enc"]
INTERACTION_FEATURES = ["PM_ratio", "PM_sum", "NO2_CO", "Temp_x_Hum"]
BASE_POLLUTANTS      = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                        "O3", "Benzene", "Toluene", "Xylene"]
METEOROLOGICAL       = ["Temp_Mean", "Humidity_Mean", "Wind_Speed_Mean", "Wind_Dir_Mean"]

BASE_POLLUTANTS = [c for c in BASE_POLLUTANTS if c in all_features]
METEOROLOGICAL  = [c for c in METEOROLOGICAL  if c in all_features]

feature_groups = {
    "Lag Features":        LAG_FEATURES,
    "Rolling Statistics":  ROLLING_FEATURES,
    "Time Features":       TIME_FEATURES,
    "City Encoding":       CITY_FEATURES,
    "Interaction Features":INTERACTION_FEATURES,
    "Base Pollutants":     BASE_POLLUTANTS,
    "Meteorological":      METEOROLOGICAL,
}

for name, features in feature_groups.items():
    print(f"  {name:25s}: {len(features):3d} features")
print(f"\n  Total features: {len(all_features)}")

# ─────────────────────────────────────────────
# STEP 4: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
SPLIT_DATE = "2020-01-01"
train_df = df[df["Date"] < SPLIT_DATE].copy()
test_df  = df[df["Date"] >= SPLIT_DATE].copy()

print(f"\n  Train: {len(train_df)} rows")
print(f"  Test:  {len(test_df)} rows")

# ─────────────────────────────────────────────
# STEP 5: MODEL TRAINING UTILITY
# ─────────────────────────────────────────────
ALL_MODELS = ["Ridge", "XGBoost", "LightGBM", "RandomForest"]

def train_and_evaluate(feature_cols, model_type="XGBoost", name="Baseline"):
    """Train model with given features and return metrics."""
    X_train = train_df[feature_cols].values
    y_train = train_df["AQI"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["AQI"].values

    if model_type == "Ridge":
        scaler_X = MinMaxScaler()
        X_train  = scaler_X.fit_transform(X_train)
        X_test   = scaler_X.transform(X_test)
        model    = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

    elif model_type == "LightGBM":
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            max_depth=8, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train)

    elif model_type == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features=0.6, n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y_pred = model.predict(X_test)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)

    return {
        "Experiment": name,
        "Model":      model_type,
        "Features":   len(feature_cols),
        "MSE":        mse,
        "RMSE":       rmse,
        "R²":         r2,
    }

# ─────────────────────────────────────────────
# STEP 6: ABLATION EXPERIMENTS
# ─────────────────────────────────────────────
ablation_results = []

# ── Experiment 1: Baseline (All Features) — all models ──
print("\n" + "=" * 60)
print("[1] Baseline: All Features — all models")
print("=" * 60)

for model_type in ALL_MODELS:
    result = train_and_evaluate(all_features, model_type, "Baseline (All)")
    ablation_results.append(result)
    print(f"  {model_type:15s}  MSE={result['MSE']:8.3f}  RMSE={result['RMSE']:7.3f}  R²={result['R²']:.6f}")

# ── Experiment 2: Remove Each Feature Group — all models ──
print("\n" + "=" * 60)
print("[2] Removing each feature group (one at a time) — all models")
print("=" * 60)

for group_name, group_features in feature_groups.items():
    if len(group_features) == 0:
        continue

    remaining_features = [f for f in all_features if f not in group_features]
    print(f"\n[Ablation] Removing: {group_name} ({len(group_features)} features)")
    print(f"  Remaining: {len(remaining_features)} features")

    for model_type in ALL_MODELS:
        result = train_and_evaluate(
            remaining_features, model_type, f"Without {group_name}"
        )
        ablation_results.append(result)
        print(f"  {model_type:15s}  MSE={result['MSE']:8.3f}  RMSE={result['RMSE']:7.3f}  R²={result['R²']:.6f}")

# ── Experiment 3: Only Each Feature Group — all models ──
print("\n" + "=" * 60)
print("[3] Using ONLY each feature group (+ Base Pollutants) — all models")
print("=" * 60)

for group_name, group_features in feature_groups.items():
    if len(group_features) == 0 or group_name == "Base Pollutants":
        continue

    only_features = list(set(group_features + BASE_POLLUTANTS))
    if len(only_features) < 5:
        continue

    print(f"\n[Only] Using: {group_name} + Base Pollutants ({len(only_features)} features)")

    for model_type in ALL_MODELS:
        result = train_and_evaluate(
            only_features, model_type, f"Only {group_name}"
        )
        ablation_results.append(result)
        print(f"  {model_type:15s}  MSE={result['MSE']:8.3f}  RMSE={result['RMSE']:7.3f}  R²={result['R²']:.6f}")

# ── Experiment 4: Cumulative Addition — all models ──
print("\n" + "=" * 60)
print("[4] Cumulative Addition (starting from base pollutants) — all models")
print("=" * 60)

cumulative_features = BASE_POLLUTANTS.copy()
addition_order = [
    ("Base Only",       []),
    ("+ City",          CITY_FEATURES),
    ("+ Time",          TIME_FEATURES),
    ("+ Lag",           LAG_FEATURES),
    ("+ Rolling",       ROLLING_FEATURES),
    ("+ Interactions",  INTERACTION_FEATURES),
]

for stage_name, new_features in addition_order:
    if stage_name != "Base Only":
        cumulative_features = list(set(cumulative_features + new_features))

    print(f"\n[Cumulative] {stage_name}: {len(cumulative_features)} features")

    for model_type in ALL_MODELS:
        result = train_and_evaluate(cumulative_features, model_type, stage_name)
        ablation_results.append(result)
        print(f"  {model_type:15s}  MSE={result['MSE']:8.3f}  RMSE={result['RMSE']:7.3f}  R²={result['R²']:.6f}")

# ─────────────────────────────────────────────
# STEP 7: SAVE RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(ablation_results)
results_df.to_csv("ablation_study_results.csv", index=False)
print("\n  Saved: ablation_study_results.csv")
print("\n" + results_df.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 8: VISUALISATIONS
# ─────────────────────────────────────────────
MODEL_COLORS = {
    "Ridge":        "#AB47BC",   # purple
    "XGBoost":      "#2196F3",   # blue
    "LightGBM":     "#4CAF50",   # green
    "RandomForest": "#FF9800",   # orange
}

# ──────────────────────────────────────────────────────────
# PLOT SET 1 — Four separate ablation_feature_importance
#              files, one per model
# ──────────────────────────────────────────────────────────
for model_type in ALL_MODELS:
    color = MODEL_COLORS[model_type]

    baseline_mse = results_df[
        (results_df["Experiment"] == "Baseline (All)") &
        (results_df["Model"] == model_type)
    ]["MSE"].values[0]

    baseline_r2 = results_df[
        (results_df["Experiment"] == "Baseline (All)") &
        (results_df["Model"] == model_type)
    ]["R²"].values[0]

    removal = results_df[
        results_df["Experiment"].str.contains("Without") &
        (results_df["Model"] == model_type)
    ].copy()

    removal["MSE_Increase"]   = removal["MSE"] - baseline_mse
    removal["MSE_Increase_%"] = (removal["MSE_Increase"] / baseline_mse) * 100
    removal["Feature_Group"]  = removal["Experiment"].str.replace("Without ", "")
    removal = removal.sort_values("MSE_Increase", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Feature Group Ablation Study — {model_type}",
        fontsize=14, fontweight="bold"
    )

    # Left: MSE increase when group removed
    bar_colors = ["#E57373" if x > 0 else "#81C784" for x in removal["MSE_Increase"]]
    axes[0].barh(removal["Feature_Group"], removal["MSE_Increase"], color=bar_colors)
    axes[0].set_xlabel("MSE Increase (higher = more important)")
    axes[0].set_title("Impact of Removing Each Feature Group")
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.4)
    axes[0].grid(axis="x", alpha=0.3)

    # Annotate percentage labels
    for bar, pct in zip(axes[0].patches, removal["MSE_Increase_%"]):
        x_pos = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        sign  = "+" if pct >= 0 else ""
        axes[0].text(
            x_pos + abs(removal["MSE_Increase"].max()) * 0.01,
            y_pos,
            f"{sign}{pct:.1f}%",
            va="center", fontsize=8
        )

    # Right: R² after removing each group
    removal_r2 = results_df[
        results_df["Experiment"].str.contains("Without") &
        (results_df["Model"] == model_type)
    ].copy()
    removal_r2["Feature_Group"] = removal_r2["Experiment"].str.replace("Without ", "")
    removal_r2 = removal_r2.sort_values("R²", ascending=True)

    axes[1].barh(removal_r2["Feature_Group"], removal_r2["R²"], color=color, alpha=0.8)
    axes[1].axvline(
        x=baseline_r2, color="red", linestyle="--",
        label=f"Baseline R²={baseline_r2:.4f}"
    )
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("R² After Removing Each Feature Group")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fname = f"ablation_feature_importance_{model_type.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ──────────────────────────────────────────────────────────
# PLOT SET 2 — Combined file: cumulative + model comparison
# ──────────────────────────────────────────────────────────
cumulative_stages = [s for s, _ in addition_order]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(
    "Cumulative Feature Addition & Model Comparison — All Models",
    fontsize=15, fontweight="bold"
)

# ── Top-left: MSE by cumulative stage (all models) ──
ax = axes[0, 0]
for model_type in ALL_MODELS:
    cum_data = results_df[
        results_df["Experiment"].isin(cumulative_stages) &
        (results_df["Model"] == model_type)
    ].copy()
    # Preserve the addition order
    cum_data["_order"] = cum_data["Experiment"].map(
        {s: i for i, s in enumerate(cumulative_stages)}
    )
    cum_data = cum_data.sort_values("_order")
    ax.plot(
        range(len(cum_data)), cum_data["MSE"],
        marker="o", linewidth=2, markersize=7,
        label=model_type, color=MODEL_COLORS[model_type]
    )
ax.set_xticks(range(len(cumulative_stages)))
ax.set_xticklabels(cumulative_stages, rotation=35, ha="right")
ax.set_ylabel("MSE")
ax.set_title("Cumulative Feature Addition — MSE")
ax.legend()
ax.grid(alpha=0.3)

# ── Top-right: R² by cumulative stage (all models) ──
ax = axes[0, 1]
for model_type in ALL_MODELS:
    cum_data = results_df[
        results_df["Experiment"].isin(cumulative_stages) &
        (results_df["Model"] == model_type)
    ].copy()
    cum_data["_order"] = cum_data["Experiment"].map(
        {s: i for i, s in enumerate(cumulative_stages)}
    )
    cum_data = cum_data.sort_values("_order")
    ax.plot(
        range(len(cum_data)), cum_data["R²"],
        marker="s", linewidth=2, markersize=7,
        label=model_type, color=MODEL_COLORS[model_type]
    )
ax.set_xticks(range(len(cumulative_stages)))
ax.set_xticklabels(cumulative_stages, rotation=35, ha="right")
ax.set_ylabel("R²")
ax.set_title("Cumulative Feature Addition — R²")
ax.legend()
ax.grid(alpha=0.3)

# ── Bottom-left: Grouped bar — MSE for key ablation experiments ──
key_experiments = [
    "Baseline (All)",
    "Without Lag Features",
    "Without Rolling Statistics",
    "Without Time Features",
    "Without Base Pollutants",
    "Without Meteorological",
]

ax = axes[1, 0]
x     = np.arange(len(key_experiments))
width = 0.2
for i, model_type in enumerate(ALL_MODELS):
    vals = []
    for exp in key_experiments:
        row = results_df[
            (results_df["Experiment"] == exp) &
            (results_df["Model"] == model_type)
        ]
        vals.append(row["MSE"].values[0] if len(row) > 0 else 0)
    ax.bar(
        x + i * width, vals, width,
        label=model_type, color=MODEL_COLORS[model_type], alpha=0.85
    )
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(key_experiments, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("MSE")
ax.set_title("MSE Comparison — Key Ablation Experiments")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# ── Bottom-right: Grouped bar — R² for key ablation experiments ──
ax = axes[1, 1]
for i, model_type in enumerate(ALL_MODELS):
    vals = []
    for exp in key_experiments:
        row = results_df[
            (results_df["Experiment"] == exp) &
            (results_df["Model"] == model_type)
        ]
        vals.append(row["R²"].values[0] if len(row) > 0 else 0)
    ax.bar(
        x + i * width, vals, width,
        label=model_type, color=MODEL_COLORS[model_type], alpha=0.85
    )
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(key_experiments, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("R²")
ax.set_title("R² Comparison — Key Ablation Experiments")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("ablation_cumulative_and_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ablation_cumulative_and_comparison.png")

# ─────────────────────────────────────────────
# STEP 9: KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS FROM ABLATION STUDY")
print("=" * 60)

for model_type in ALL_MODELS:
    baseline_mse = results_df[
        (results_df["Experiment"] == "Baseline (All)") &
        (results_df["Model"] == model_type)
    ]["MSE"].values[0]

    removal = results_df[
        results_df["Experiment"].str.contains("Without") &
        (results_df["Model"] == model_type)
    ].copy()
    removal["MSE_Increase"]   = removal["MSE"] - baseline_mse
    removal["Feature_Group"]  = removal["Experiment"].str.replace("Without ", "")
    removal = removal.sort_values("MSE_Increase", ascending=False)

    most   = removal.iloc[0]
    least  = removal.iloc[-1]

    cum_data = results_df[
        results_df["Experiment"].isin(cumulative_stages) &
        (results_df["Model"] == model_type)
    ].copy()
    cum_data["_order"] = cum_data["Experiment"].map(
        {s: i for i, s in enumerate(cumulative_stages)}
    )
    cum_data = cum_data.sort_values("_order")

    base_mse  = cum_data.iloc[0]["MSE"]
    final_mse = cum_data.iloc[-1]["MSE"]
    improvement = ((base_mse - final_mse) / base_mse) * 100

    print(f"\n  ── {model_type} ──")
    print(f"  Most Important : {most['Feature_Group']}"
          f"  (MSE +{most['MSE_Increase']:.2f} when removed)")
    print(f"  Least Important: {least['Feature_Group']}"
          f"  (MSE {least['MSE_Increase']:+.2f} when removed)")
    print(f"  Cumulative gain: {base_mse:.2f} → {final_mse:.2f}  ({improvement:.1f}% improvement)")

print("\n" + "=" * 60)
print("ABLATION STUDY COMPLETE!")
print("=" * 60)