import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
from requests.auth import HTTPBasicAuth
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from joblib import dump

# ==============================
# 1. Fetch Data
# ==============================
url = "https://cbpatelcbri.pythonanywhere.com/get-master-json"
response = requests.get(url, auth=HTTPBasicAuth("CBPATEL", "8120410372"))
response.raise_for_status()
data = response.json()
if "data" not in data:
    raise Exception("'data' key not found in response")

df = pd.DataFrame(data["data"])
original_len = len(df)

# ==============================
# 2. Filter Required Houses
# ==============================
houses_wanted = ['B-21', 'B-22', 'B-33', 'IHBT-31', 'IHBT-30', 'MP01']
df = df[df['Home Number'].isin(houses_wanted)].reset_index(drop=True)
print(f"✅ Filtered dataset for houses: {houses_wanted}")
print(f"Total records after filtering: {len(df)} / {original_len}")

# ==============================
# 3. Preprocess & Clean Data
# ==============================
df['original_date'] = df.get('Date')
df['original_time'] = df.get('Time')

def parse_datetime(row):
    try:
        return parser.parse(f"{row['Date']} {row['Time']}", dayfirst=True)
    except:
        try:
            return parser.parse(str(row['Date']), dayfirst=True)
        except:
            return pd.NaT

df['datetime'] = df.apply(parse_datetime, axis=1)
df['datetime'] = df['datetime'].fillna(pd.Timestamp.now())
df = df.sort_values(['Home Number', 'datetime']).reset_index(drop=True)

# Convert numeric fields
df['Reading'] = pd.to_numeric(df['Reading'], errors='coerce')
df['Reading'] = df.groupby('Home Number')['Reading'].transform(lambda x: x.ffill().bfill())
df['Temperature'] = pd.to_numeric(df.get('Temperature', 0), errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(0)
df['Humidity'] = pd.to_numeric(df.get('Humidity', 0), errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(0)

# Remove faulty statuses
faulty_status = {"Error", "Disconnected", "Power Failure"}
if "Status" in df.columns:
    df['Status'] = df['Status'].astype(str)
    df = df[~df['Status'].isin(faulty_status)]

# Compute per-house energy (difference between consecutive readings)
df['energy'] = df.groupby('Home Number')['Reading'].diff().fillna(0)

# Remove outliers (3*IQR rule)
def remove_outliers(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return series.clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

df['energy'] = remove_outliers(df['energy'])

# Handle long zero stretches
zero_indices_to_nan = []
for home, group in df.groupby("Home Number"):
    is_zero = group['energy'] == 0
    grp = is_zero.groupby((~is_zero).cumsum())
    for _, run in grp:
        if run.all() and run.sum() >= 10:
            zero_indices_to_nan.extend(run.index.tolist())
df.loc[zero_indices_to_nan, 'energy'] = np.nan

df['energy'] = df.groupby('Home Number')['energy'].transform(lambda x: x.interpolate(method='linear', limit_direction='both')).fillna(0)
df['energy'] = df.groupby('Home Number')['energy'].transform(lambda x: x.rolling(3, min_periods=1).mean())

cleaned_len = len(df)
print(f"Original rows: {original_len}, After cleaning: {cleaned_len}")

# ==============================
# 4. Feature Engineering
# ==============================
df['is_weekday'] = (df['datetime'].dt.weekday < 5).astype(int)
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['month'] = df['datetime'].dt.month

region_map = {"B-21": "Roorkee", "B-22": "Roorkee", "B-33": "Roorkee",
              "IHBT-30": "Himachal", "IHBT-31": "Himachal", "MP01": "MadhyaPradesh"}
df['region'] = df['Home Number'].map(region_map).fillna("Unknown")
region_dummies = pd.get_dummies(df['region'], prefix="region")
df = pd.concat([df, region_dummies], axis=1)

# Lag & rolling features
for lag in [1, 2, 3, 24]:
    df[f'lag_{lag}'] = df.groupby('Home Number')['energy'].shift(lag)

for window in [3, 7]:
    df[f'rolling_mean_{window}'] = df.groupby('Home Number')['energy'].shift(1).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'rolling_std_{window}'] = df.groupby('Home Number')['energy'].shift(1).rolling(window, min_periods=1).std().reset_index(level=0, drop=True)

df['diff_1'] = df.groupby('Home Number')['energy'].diff(1)
df['diff_7'] = df.groupby('Home Number')['energy'].diff(7)
df['humidity_temp'] = df['Humidity'] * df['Temperature']
df['region_temp'] = df['Temperature'] * df['region_Himachal'].fillna(0)
df['region_weekday'] = df['is_weekday'] * df['region_Roorkee'].fillna(0)

features = [
    'lag_1', 'lag_2', 'lag_3', 'lag_24',
    'rolling_mean_3', 'rolling_mean_7',
    'rolling_std_3', 'rolling_std_7',
    'Humidity', 'Temperature',
    'humidity_temp', 'region_temp', 'region_weekday',
    'is_weekday', 'hour', 'day_of_week', 'month',
    'diff_1', 'diff_7'
] + list(region_dummies.columns)

# ==============================
# 5. Train Per-Home Models + Plots
# ==============================
os.makedirs("models", exist_ok=True)
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

house_results = {}
all_houses_summary = []

for home, group in df.groupby("Home Number"):
    group = group.dropna(subset=['energy'] + features)
    if len(group) < 50:
        print(f"⚠️ Skipping {home}: insufficient data ({len(group)} rows)")
        continue

    print(f"\n🏠 Training model for Home: {home}")
    X = group[features].copy()
    y = group['energy']
    split_idx = int(len(group) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
        'max_samples': [0.7, 0.8, None]
    }

    rf = RandomForestRegressor(random_state=42)
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=3,
                                     scoring='r2', random_state=42, n_jobs=-1)
    rand_search.fit(X_train, y_train)
    best_params = rand_search.best_params_
    print(f"🔹 RandomizedSearch Best Params for {home}: {best_params}")

    # GridSearch refinement
    refined_grid = {
        'n_estimators': [max(50, best_params['n_estimators'] - 50),
                         best_params['n_estimators'],
                         best_params['n_estimators'] + 50],
        'max_depth': [best_params['max_depth'],
                      (best_params['max_depth'] + 5) if best_params['max_depth'] else None],
        'min_samples_split': [best_params['min_samples_split']],
        'min_samples_leaf': [best_params['min_samples_leaf']],
        'max_features': [best_params['max_features']],
        'max_samples': [0.7, 0.8]
    }

    grid_search = GridSearchCV(rf, param_grid=refined_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"🔹 GridSearch Best Params for {home}: {grid_search.best_params_}")

    dump(best_model, f"models/model_home_{home}.pkl")

    # Predictions & metrics
    y_pred = best_model.predict(X_test)
    dates_test = group['datetime'].iloc[split_idx:]
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    house_results[home] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"✅ Home {home} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # ---- Feature Importances ----
    fi = dict(zip(X.columns, best_model.feature_importances_))
    mi = dict(zip(X.columns, mutual_info_regression(X, y, discrete_features=False, random_state=42)))
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_dict = dict(zip(X.columns, perm.importances_mean))

    house_df = pd.DataFrame({
        'Feature': X.columns,
        'Feature_Importance': [fi[f] for f in X.columns],
        'Mutual_Info': [mi[f] for f in X.columns],
        'Permutation_Importance': [perm_dict[f] for f in X.columns]
    }).sort_values('Feature_Importance', ascending=False)
    house_df['Home'] = home
    all_houses_summary.append(house_df)

    # ==========================
    # 📈 Save Plots per Home
    # ==========================
    # Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(f"Home {home} - Energy Prediction")
    plt.xlabel("Test Samples")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_actual_vs_pred.png"))
    plt.close()

    # Residuals
    residuals = y_test.values - y_pred
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, color='red', label='Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Home {home} - Prediction Residuals")
    plt.xlabel("Test Samples")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_residuals.png"))
    plt.close()

    # Cumulative Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, np.cumsum(y_test.values), label='Actual', color='blue')
    plt.plot(dates_test, np.cumsum(y_pred), label='Predicted', color='orange')
    plt.title(f"Home {home} - Energy Comparison")
    plt.xlabel("Time")
    plt.ylabel("Reading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_cumulative.png"))
    plt.close()

# ==============================
# 6. Summary Metrics
# ==============================
if house_results:
    metrics_df = pd.DataFrame.from_dict(house_results, orient='index').reset_index().rename(columns={'index': 'Home Number'})

    plt.figure(figsize=(10, 5))
    plt.bar(metrics_df['Home Number'], metrics_df['MAE'], label='MAE', alpha=0.7)
    plt.bar(metrics_df['Home Number'], metrics_df['RMSE'], label='RMSE', alpha=0.7)
    plt.title("Per-Home Error Metrics (MAE & RMSE)")
    plt.xticks(rotation=90)
    plt.ylabel("Error Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_home_error_metrics.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(metrics_df['R2'], metrics_df['MAE'], color='orange', label='MAE')
    plt.scatter(metrics_df['R2'], metrics_df['RMSE'], color='blue', label='RMSE')
    plt.title("Error vs Model Accuracy (R² per Home)")
    plt.xlabel("R² Score")
    plt.ylabel("Error (MAE / RMSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "r2_vs_error_scatter.png"))
    plt.close()

    print(f"\n✅ Finished training for {len(house_results)} homes.")
    print(metrics_df)
    print(f"📂 All plots saved in: {os.path.abspath(output_dir)}")
else:
    print("⚠️ No valid models were trained.")

# ==============================
# 7. Save Feature Importances
# ==============================
if all_houses_summary:
    final_summary = pd.concat(all_houses_summary, ignore_index=True)
    final_summary.to_csv("house_feature_importances_tuned_filtered.csv", index=False)
    print("\n✅ Feature importances saved to 'house_feature_importances_tuned_filtered.csv'")
else:
    print("⚠️ No valid house models trained.")
