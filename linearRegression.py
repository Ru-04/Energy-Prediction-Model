import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ==============================
# 0. Setup Fonts & Output Folder
# ==============================
plt.rcParams['font.family'] = 'DejaVu Sans'   # Compatible with all Unicode
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 1. Fetch Data from API
# ==============================
url = "https://cbpatelcbri.pythonanywhere.com/get-master-json"
response = requests.get(url, auth=HTTPBasicAuth("CBPATEL", "8120410372"))

if response.status_code != 200:
    raise Exception(f"❌ Failed to fetch data: {response.text}")

data = response.json()
if "data" not in data:
    raise Exception("❌ 'data' key not found in response")

df = pd.DataFrame(data["data"])
original_len = len(df)

# ==============================
# 2. Clean & Parse Datetime
# ==============================
df['original_date'] = df['Date']
df['original_time'] = df['Time']

def parse_datetime(row):
    try:
        return parser.parse(f"{row['Date']} {row['Time']}", dayfirst=True)
    except:
        try:
            return parser.parse(str(row['Date']), dayfirst=True)
        except:
            return pd.Timestamp.now()

df['datetime'] = df.apply(parse_datetime, axis=1)
df = df.sort_values(['Home Number','datetime'])
df['Home Number'] = df['Home Number'].fillna('Unknown')

# ==============================
# 3. Numeric Conversion & Cleaning
# ==============================
df['Reading'] = pd.to_numeric(df['Reading'], errors='coerce').ffill().bfill().fillna(0)
df['Temperature'] = pd.to_numeric(df.get('Temperature', 0), errors='coerce').ffill().bfill().fillna(0)
df['Humidity'] = pd.to_numeric(df.get('Humidity', 0), errors='coerce').ffill().bfill().fillna(0)

# Calculate Energy as difference of readings
df['energy'] = df.groupby('Home Number')['Reading'].diff().fillna(0)

# Remove outliers
def remove_outliers(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return series.clip(Q1 - 3*IQR, Q3 + 3*IQR)

df['energy'] = remove_outliers(df['energy'])

# ==============================
# 4. Feature Engineering
# ==============================
df['is_weekday'] = (df['datetime'].dt.weekday < 5).astype(int)
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.weekday
df['month'] = df['datetime'].dt.month

for lag in [1, 2, 3, 24]:
    df[f'lag_{lag}'] = df.groupby('Home Number')['energy'].shift(lag)

for window in [3, 7]:
    df[f'rolling_mean_{window}'] = df.groupby('Home Number')['energy'].shift(1).rolling(window, min_periods=1).mean()
    df[f'rolling_std_{window}'] = df.groupby('Home Number')['energy'].shift(1).rolling(window, min_periods=1).std()

df['diff_1'] = df.groupby('Home Number')['energy'].diff(1)
df['diff_7'] = df.groupby('Home Number')['energy'].diff(7)
df['humidity_temp'] = df['Humidity'] * df['Temperature']

# ==============================
# 5. Train Model per Home
# ==============================
features = [
    'lag_1','lag_2','lag_3','lag_24',
    'rolling_mean_3','rolling_mean_7',
    'rolling_std_3','rolling_std_7',
    'Humidity','Temperature','humidity_temp',
    'is_weekday','hour','day_of_week','month'
]

house_results = {}

for home, group in df.groupby('Home Number'):
    group = group.dropna(subset=['energy'] + features)
    if len(group) < 50:
        continue

    X = group[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = group['energy']

    split_idx = int(len(group) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = group['datetime'].iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    house_results[home] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # ========== PLOTS ==========
    # Actual vs Predicted
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(f"Home {home} - Energy Prediction", fontsize=14)
    plt.xlabel("Test Samples")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_actual_vs_pred.png"))
    plt.show()
    plt.close()

    # Residuals
    residuals = y_test.values - y_pred
    plt.figure(figsize=(10,5))
    plt.plot(residuals, label='Residuals (Actual - Predicted)', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Home {home} - Prediction Error (Residuals)", fontsize=14)
    plt.xlabel("Test Samples")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_residuals.png"))
    plt.show()
    plt.close()

    # Cumulative Comparison
    plt.figure(figsize=(10,5))
    plt.plot(dates_test, np.cumsum(y_test.values), label='Cumulative Actual', color='blue')
    plt.plot(dates_test, np.cumsum(y_pred), label='Cumulative Predicted', color='orange')
    plt.title(f"Home {home} - Cumulative Energy Comparison", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{home}_cumulative.png"))
    plt.show()
    plt.close()

# ==============================
# 6. Summary Metrics
# ==============================
metrics_df = pd.DataFrame.from_dict(house_results, orient='index').reset_index().rename(columns={'index':'Home Number'})

# Per-Home MAE & RMSE
plt.figure(figsize=(10,5))
plt.bar(metrics_df['Home Number'], metrics_df['MAE'], label='MAE', alpha=0.7)
plt.bar(metrics_df['Home Number'], metrics_df['RMSE'], label='RMSE', alpha=0.7)
plt.title("Per-Home Error Metrics (MAE & RMSE)", fontsize=14)
plt.xticks(rotation=90)
plt.ylabel("Error Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "per_home_error_metrics.png"))
plt.show()
plt.close()

# R² vs Error Scatter
plt.figure(figsize=(8,5))
plt.scatter(metrics_df['R2'], metrics_df['MAE'], color='orange', label='MAE')
plt.scatter(metrics_df['R2'], metrics_df['RMSE'], color='blue', label='RMSE')
plt.title("Error vs Model Accuracy (R² per Home)", fontsize=14)
plt.xlabel("R² Score")
plt.ylabel("Error (MAE / RMSE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "r2_vs_error_scatter.png"))
plt.show()
plt.close()

# ==============================
# 7. Summary Output
# ==============================
print(f"\n✅ Finished training for {len(house_results)} homes.")
print(metrics_df)
print(f"\n📂 All plots saved in: {os.path.abspath(output_dir)}")
