import logging
import requests
from requests.auth import HTTPBasicAuth
from pathlib import Path
from datetime import datetime
from dateutil import parser

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path("/content")
MODELS_DIR = BASE_DIR / "trained_models"
SCALERS_DIR = BASE_DIR / "trained_scalers"
LOG_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCALERS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "per_home_train.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# HELPERS
# ==========================================
def parse_datetime_row(row):
    try:
        return parser.parse(f"{row['Date']} {row['Time']}", dayfirst=True)
    except Exception:
        return datetime.now()

def remove_outliers(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return series.clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)

# ==========================================
# TRAIN SINGLE HOME MODEL
# ==========================================
def train_home_model(df_home, home_id):
    logging.info(f" Training model for Home: {home_id}")

    FEATURES = ["energy", "Temperature", "Humidity"]
    SEQ_LEN = 24

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_home[FEATURES])

    X, y = create_sequences(scaled, SEQ_LEN)

    if len(X) < 100:
        logging.warning(f" Not enough data for {home_id}, skipping")
        return

    model = Sequential([
        LSTM(64, input_shape=(SEQ_LEN, len(FEATURES)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Save paths
    model_path = MODELS_DIR / f"home_{home_id}"
    scaler_path = SCALERS_DIR / f"scaler_{home_id}.joblib"

    if model_path.exists():
        import shutil
        shutil.rmtree(model_path)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    logging.info(f"Saved model for {home_id}")

# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def train_all_homes():
    logging.info("Starting per-home LSTM training")

    url = "https://cbpatelcbri.pythonanywhere.com/get-master-json"
    response = requests.get(
        url,
        auth=HTTPBasicAuth("CBPATEL", "8120410372"),
        timeout=30
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json()["data"])

    df["datetime"] = df.apply(parse_datetime_row, axis=1)
    df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce").fillna(0)
    df["Temperature"] = pd.to_numeric(df.get("Temperature", 25), errors="coerce").fillna(25)
    df["Humidity"] = pd.to_numeric(df.get("Humidity", 50), errors="coerce").fillna(50)

    df = df.sort_values(by=["Home Number", "datetime"])

    df["energy"] = df.groupby("Home Number")["Reading"].diff().fillna(0)
    df["energy"] = remove_outliers(df["energy"])
    df = df[df["energy"] >= 0]

    homes = df["Home Number"].unique()
    logging.info(f" Found homes: {homes}")

    for home_id in homes:
        df_home = df[df["Home Number"] == home_id].copy()
        train_home_model(df_home, home_id)

    logging.info(" All home models trained successfully")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    train_all_homes()
