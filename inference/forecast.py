import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

import numpy as np
import pandas as pd
import joblib
import requests
import pvlib
from tensorflow.keras.models import load_model

# =====================================================
# PATHS  — artifacts séparés du one-step
# =====================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
MODELS_DIR    = os.path.join(ARTIFACTS_DIR, "models")

# =====================================================
# LOAD ARTIFACTS
# =====================================================
top_modes = np.load(
    os.path.join(BASE_DIR, "..", "top_modes.npy"),
    allow_pickle=True
).tolist()
top_modes.append("res")

feature_selection = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_selection.pkl"))
scaler_y          = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_y.pkl"))
scaler_std        = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_std.pkl"))
scaler_minmax     = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_minmax.pkl"))

# =====================================================
# LOAD MODELS
# =====================================================
models_ms = {}
for m in top_modes:
    mode_col      = "mode_res" if m == "res" else f"mode_{m}"
    models_ms[m]  = load_model(os.path.join(MODELS_DIR, f"model_{mode_col}.keras"))

# =====================================================
# CONSTANTES
# =====================================================
WINDOW           = 48
STD_FEATURES     = [
    "ambient_temperature", "wind_speed", "module_temperature",
    "ghi", "dni", "dhi", "global_radiation", "tilted_radiation",
    "temp_radiation", "zenith", "solar_elevation", "clearness_index"
]
MINMAX_FEATURES  = ["humidity", "clouds"]

# =====================================================
# WEATHER
# =====================================================
def get_weather():
    lat, lon = 30.92, -6.91
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"windspeed_10m,shortwave_radiation"
        f"&forecast_days=3"
        f"&timezone=Africa%2FCasablanca"
    )
    data = requests.get(url).json()["hourly"]
    df = pd.DataFrame({
        "timestamp":           pd.to_datetime(data["time"]),
        "ambient_temperature": data["temperature_2m"],
        "humidity":            data["relative_humidity_2m"],
        "wind_speed":          data["windspeed_10m"],
        "global_radiation":    data["shortwave_radiation"],
    })
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Africa/Casablanca")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Africa/Casablanca")
    return df

# =====================================================
# FIX TIMESERIES
# =====================================================
def fix_timeseries(df):
    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="30min",
        tz="Africa/Casablanca"
    )
    df = df.reindex(full_index)
    df = df.interpolate().ffill().bfill()
    df.index.name = "timestamp"
    return df

# =====================================================
# PREPROCESS
# =====================================================
def preprocess(df):
    location = pvlib.location.Location(
        latitude=30.92, longitude=-6.91, tz="Africa/Casablanca"
    )
    solpos = location.get_solarposition(df.index)
    df["zenith"]          = solpos["zenith"]
    df["solar_elevation"] = solpos["elevation"]

    cs = location.get_clearsky(df.index)
    df["ghi"] = cs["ghi"]
    df["dni"] = cs["dni"]
    df["dhi"] = cs["dhi"]

    df["clouds"] = np.clip(
        100 * (1 - df["global_radiation"] / (df["ghi"] + 1e-6)), 0, 100
    )
    df.loc[df["solar_elevation"] <= 0, "global_radiation"] = 0

    df["clearness_index"] = np.clip(
        df["global_radiation"] / (df["ghi"] + 1e-6), 0, 1.2
    )

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25, surface_azimuth=180,
        dni=df["dni"], ghi=df["global_radiation"], dhi=df["dhi"],
        solar_zenith=df["zenith"], solar_azimuth=solpos["azimuth"]
    )
    df["tilted_radiation"]   = poa["poa_global"]
    df["module_temperature"] = df["ambient_temperature"] + 0.03 * df["tilted_radiation"]

    hour_float     = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)

    df["temp_radiation"]  = df["module_temperature"] * df["tilted_radiation"]
    df["is_day"]          = (df["solar_elevation"] > 0).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate().ffill().fillna(0)
    return df

# =====================================================
# SCALE
# =====================================================
def scale_features(df):
    is_day_raw          = df["is_day"].values.copy()
    df[STD_FEATURES]    = scaler_std.transform(df[STD_FEATURES])
    df[MINMAX_FEATURES] = scaler_minmax.transform(df[MINMAX_FEATURES])
    df["is_day"]        = is_day_raw
    return df

# =====================================================
# PREDICT — retourne array 1D (N pas)
# =====================================================
def _predict(df_scaled, now_idx):
    start_idx = max(0, now_idx - WINDOW + 1)
    df_window = df_scaled.iloc[start_idx : now_idx + 1].copy()

    if len(df_window) < WINDOW:
        pad       = pd.DataFrame(
            np.zeros((WINDOW - len(df_window), df_window.shape[1])),
            columns=df_window.columns
        )
        df_window = pd.concat([pad, df_window], ignore_index=True)

    predictions = []
    for m in top_modes:
        mode_col = "mode_res" if m == "res" else f"mode_{m}"
        features = feature_selection.get(mode_col, [])
        features = [f for f in features if f in df_window.columns]
        if not features:
            continue
        X    = df_window[features].values.reshape(1, WINDOW, len(features))
        pred = models_ms[m].predict(X, verbose=0)
        predictions.append(pred)

    if not predictions:
        raise ValueError("Aucune prédiction produite")

    final = np.sum(predictions, axis=0)
    pv    = scaler_y.inverse_transform(final.reshape(-1, 1)).flatten()[0]
    return float(pv)

# =====================================================
# FORECAST 12H — retourne liste de dicts
# =====================================================
def forecast_12h():

    now = pd.Timestamp.now(tz="Africa/Casablanca")

    df = get_weather()
    df = fix_timeseries(df)
    df = preprocess(df)

    df_scaled = scale_features(df.copy())

    now_idx = df_scaled.index.get_indexer(
        [now],
        method="nearest"
    )[0]

    print("\nCURRENT TIME:")
    print(now)

    print("\nNEAREST TIMESTAMP USED:")
    print(df_scaled.index[now_idx])

    # fenêtre utilisée
    start_idx = max(0, now_idx - WINDOW + 1)

    df_window = df_scaled.iloc[
        start_idx : now_idx + 1
    ].copy()

    if len(df_window) < WINDOW:

        pad = pd.DataFrame(
            np.zeros(
                (
                    WINDOW - len(df_window),
                    df_window.shape[1]
                )
            ),
            columns=df_window.columns
        )

        df_window = pd.concat(
            [pad, df_window],
            ignore_index=True
        )

    predictions = []

    for m in top_modes:

        mode_col = (
            "mode_res"
            if m == "res"
            else f"mode_{m}"
        )

        features = feature_selection.get(
            mode_col,
            []
        )

        features = [
            f for f in features
            if f in df_window.columns
        ]

        if not features:
            continue

        X = (
            df_window[features]
            .values
            .reshape(
                1,
                WINDOW,
                len(features)
            )
        )

        pred = models_ms[m].predict(
            X,
            verbose=0
        )

        predictions.append(pred)

    if not predictions:
        raise ValueError(
            "Aucune prédiction produite"
        )

    final_prediction = np.sum(
        predictions,
        axis=0
    )

    final_prediction = scaler_y.inverse_transform(
        final_prediction.reshape(-1, 1)
    ).flatten()

    final_prediction = np.clip(
        final_prediction,
        0,
        None
    )

    results = []
    
    for i, pv in enumerate(final_prediction):

        target_idx = now_idx + i + 1

        if target_idx >= len(df):
            break

        elev = float(
            df["solar_elevation"]
            .iloc[target_idx]
        )

        if elev <= 0:
            pv = 0.0

        ts = df.index[target_idx]

        results.append({
            "time": ts.strftime("%H:%M"),
            "pv_kw": round(float(pv), 3)
        })

    return results


# =====================================================
# TEST DIRECT
# =====================================================
if __name__ == "__main__":
    points = forecast_12h()
    for p in points:
        print(f"{p['time']}  →  {p['pv_kw']:.3f} kW")
