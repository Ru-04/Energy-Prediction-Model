import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from dateutil import parser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================
# Fetch Data
# ==============================
url = "https://cbpatelcbri.pythonanywhere.com/get-master-json"
response = requests.get(url, auth=HTTPBasicAuth("CBPATEL", "8120410372"))

if response.status_code == 200:
    data = response.json()

    if "data" in data:
        df = pd.DataFrame(data["data"])
        print(f"✅ Total rows fetched: {len(df)}")

        # ==============================
        # Clean & Prepare
        # ==============================
        df["original_date"] = df["Date"]
        df["original_time"] = df["Time"]

        # Robust datetime parsing
        def parse_datetime(row):
            try:
                return parser.parse(f"{row['Date']} {row['Time']}", dayfirst=True)
            except:
                try:
                    return parser.parse(str(row["Date"]), dayfirst=True)
                except:
                    return None

        df["datetime"] = df.apply(parse_datetime, axis=1)
        print(f"❌ Failed datetime parses: {df['datetime'].isna().sum()}")

        # Fill missing datetime as fallback
        df["datetime"] = df["datetime"].fillna(pd.Timestamp.now())
        df = df.sort_values(["Home Number", "datetime"])

        # Clean Home Number
        df["Home Number"] = df["Home Number"].fillna("Unknown")

        # Convert reading safely
        def extract_first_number(x):
            if pd.isna(x):
                return np.nan
            import re
            match = re.search(r"\d+(\.\d+)?", str(x))
            return float(match.group()) if match else np.nan

        df["Reading"] = df["Reading"].apply(extract_first_number)
        df["Temperature"] = df["Temperature"].apply(extract_first_number)
        df["Humidity"] = df["Humidity"].apply(extract_first_number)

        # Forward fill missing readings
        df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce").fillna(method="ffill").fillna(0)

        # Remove reading outliers (optional)
        def remove_outliers(series):
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            return series.clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

        df["Reading"] = remove_outliers(df["Reading"])

        houses_to_plot = ["B-21", "B-22", "B-33"]

        # ==============================
        # Compute Daily Consumption
        # ==============================
        for house in houses_to_plot:
            house_data = df[df["Home Number"] == house].copy()
            if house_data.empty:
                print(f"⚠ No data for {house}")
                continue

            print(f"🏠 Processing {house} → {len(house_data)} readings")

            # Calculate daily diff
            house_data["date_only"] = house_data["datetime"].dt.date
            house_data = house_data.sort_values("datetime")

            daily = (
                house_data.groupby("date_only", as_index=False)
                .agg({
                    "Reading": lambda x: x.max() - x.min(),
                    "Temperature": "mean",
                    "Humidity": "mean"
                })
                .rename(columns={
                    "date_only": "Date",
                    "Reading": "Daily_Consumption"
                })
            )

            # Keep Jan–Oct only
            daily["Date"] = pd.to_datetime(daily["Date"])
            daily = daily[daily["Date"].dt.month <= 10]

            # Remove invalid negatives
            daily = daily[daily["Daily_Consumption"] > 0]

            # Sort chronologically
            daily = daily.sort_values("Date")

            if daily.empty:
                print(f"⚠ No valid daily data for {house}")
                continue

            # Downsample only if too dense
            if len(daily) > 400:
                daily = daily.set_index("Date").resample("10D").mean().reset_index()
                print(f"📉 Downsampled to every 10 days for {house}")

            # ==============================
            # Plot Jan–Oct Range
            # ==============================
            fig, ax1 = plt.subplots(figsize=(14, 6))

            # Temperature line
            ax1.plot(
                daily["Date"],
                daily["Temperature"],
                color="blue",
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="Avg Temperature"
            )
            ax1.set_ylabel("Temperature (°C)", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            # Energy bars
            ax2 = ax1.twinx()
            ax2.bar(
                daily["Date"],
                daily["Daily_Consumption"],
                color="orange",
                alpha=0.4,
                label="Daily Energy Consumption"
            )
            ax2.set_ylabel("Energy Consumption (kWh)", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")
            ax2.set_ylim(0, 20)


            # Format X-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)

            plt.title(f"Daily Temperature & Energy Consumption - {house} (Jan to Oct)")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    else:
        print("❌ 'data' key not found")
else:
    print("❌ Failed to fetch data:", response.status_code)
