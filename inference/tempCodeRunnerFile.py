import os
import numpy as np
import pandas as pd
import joblib
import pvlib
import requests
from tensorflow.keras.models import load_model

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS = os.path.join(BASE_DIR, "..", "artifacts_app")

MODELS_DIR = os.path.join(ARTIFACTS, "models")

# =====================================================
# LOAD ARTIFACTS
# =====================================================

top_modes = np.load(
    os.path.join(BASE_DIR, "..", "top_modes.npy"),
    allow_pickle=True
).tolist()

top_modes.append("res")

feature_selection = joblib.load(
    os.path.join(ARTIFACTS, "feature_selection.pkl")
)

scaler_y      = joblib.load(os.path.join(ARTIFACTS, "scaler_y.pkl"))
scaler_std    = joblib.load(os.path.join(ARTIFACTS, "scaler_std.pkl"))
scaler_minmax = joblib.load(os.path.join(ARTIFACTS, "scaler_minmax.pkl"))

# =====================================================
# LOAD MODELS
# =====================================================

models = {}

for m in top_modes:
    mode_col = "mode_res" if m == "res" else f"mode_{m}"
    models[m] = load_model(
        os.path.join(MODELS_DIR, f"model_{mode_col}.keras")
    )

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
        f"&forecast_days=2"
        f"&timezone=Africa/Casablanca"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("OPEN METEO API ERROR")

    hourly = response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp":           hourly["time"],
        "ambient_temperature": hourly["temperature_2m"],
        "humidity":            hourly["relative_humidity_2m"],
        "wind_speed":          hourly["windspeed_10m"],
        "global_radiation":    hourly["shortwave_radiation"],
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = pd.to_datetime(
    df["timestamp"]).dt.tz_localize(None)
    if len(df) == 0:
        raise ValueError("Weather dataframe is empty")

    return df

# =====================================================
# FIX TIMESERIES
# =====================================================

def fix_timeseries(df):

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    # Open-Meteo retourne des pas HORAIRES.
    # resample("30min") + interpolate crée les demi-heures manquantes.
    df = df.resample("30min").interpolate(method="time")

    df = df.ffill()
    df = df.bfill()

    return df

# =====================================================
# PREPROCESS
# =====================================================

def preprocess(df):

    location = pvlib.location.Location(
        latitude=30.92,
        longitude=-6.91,
        tz="Africa/Casablanca"
    )

    solpos = location.get_solarposition(df.index)

    df["zenith"]          = solpos["zenith"]
    df["solar_elevation"] = solpos["elevation"]

    cs = location.get_clearsky(df.index)

    df["ghi"] = cs["ghi"]
    df["dni"] = cs["dni"]
    df["dhi"] = cs["dhi"]

    # CLOUDS
    df["clouds"] = np.clip(
        100 * (1 - df["global_radiation"] / (df["ghi"] + 1e-6)),
        0, 100
    )

    # NIGHT FILTER
    df.loc[df["solar_elevation"] <= 0, "global_radiation"] = 0

    # TILTED RADIATION
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        dni=df["dni"],
        ghi=df["global_radiation"],
        dhi=df["dhi"],
        solar_zenith=df["zenith"],
        solar_azimuth=solpos["azimuth"]
    )
    df["tilted_radiation"] = poa["poa_global"]

    # MODULE TEMPERATURE
    df["module_temperature"] = (
        df["ambient_temperature"] + 0.03 * df["tilted_radiation"]
    )

    # CYCLICAL FEATURES
    hour_float = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)

    # EXTRA FEATURES
    df["temp_radiation"] = df["module_temperature"] * df["tilted_radiation"]

    df["clearness_index"] = np.clip(
        df["global_radiation"] / (df["ghi"] + 1e-6),
        0, 1.2
    )

    # is_day calculé sur valeur réelle AVANT scaling
    df["is_day"] = (df["solar_elevation"] > 0).astype(int)

    # CLEAN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate()
    df = df.ffill()
    df = df.fillna(0)

    return df

# =====================================================
# SCALING
# =====================================================

def scale_features(df):

    # Sauvegarder is_day AVANT scaling (binaire, non scalé)
    is_day_raw = df["is_day"].copy()

    std_cols    = scaler_std.feature_names_in_
    minmax_cols = scaler_minmax.feature_names_in_

    df[std_cols] = scaler_std.transform(
        df.reindex(columns=std_cols, fill_value=0)
    )

    df[minmax_cols] = scaler_minmax.transform(
        df.reindex(columns=minmax_cols, fill_value=0)
    )

    # Remettre is_day non scalé pour le night fix
    df["is_day"] = is_day_raw.values

    return df

# =====================================================
# FORECAST ONE-STEP
# =====================================================

WINDOW = 48

def forecast_one_step():

    df = get_weather()
    df = fix_timeseries(df)
    now = pd.Timestamp.now()
    df = df[df.index <= now]
    df = preprocess(df)
    df = scale_features(df)

    if len(df) < WINDOW:
        raise ValueError(f"NOT ENOUGH DATA: {len(df)} < {WINDOW}")
    print(df.index.min())
    print(df.index.max())
    df_window = df.tail(WINDOW)

    last = df_window.iloc[-1]

    # DEBUG
    print(f"\n[DEBUG] Dernier pas : {df_window.index[-1]}")
    print(f"  is_day           = {last['is_day']}")
    print(f"  solar_elevation  = {last['solar_elevation']:.4f}  (scalé)")
    print(f"  global_radiation = {last['global_radiation']:.4f}  (scalé)")

    predictions = []

    for m in top_modes:

        mode_col = "mode_res" if m == "res" else f"mode_{m}"

        features = feature_selection.get(mode_col, [])
        features = [f for f in features if f in df_window.columns]

        if not features:
            print(f"[WARN] aucune feature pour {mode_col}, ignoré")
            continue

        X = df_window[features].values.reshape(1, WINDOW, len(features))
        print(mode_col,pred.flatten()[0])
        pred = models[m].predict(X, verbose=0)
        predictions.append(pred)

    if not predictions:
        raise ValueError("NO PREDICTIONS GENERATED")

    final_prediction = np.sum(predictions, axis=0)

    pv = float(
        scaler_y.inverse_transform(
            final_prediction.reshape(-1, 1)
        ).flatten()[0]
    )
    pv = max(pv, 0.0)

    print(f"[DEBUG] Prédiction brute (avant night fix) : {pv:.4f} kW")

    # NIGHT FIX — utiliser is_day (non scalé, 0 ou 1)
    if last["is_day"] == 0:
        print("[DEBUG] Night fix appliqué → pv = 0")
        pv = 0.0

    timestamp = (
        pd.Timestamp.now(tz="Africa/Casablanca")
        + pd.Timedelta(minutes=30)
    )

    return timestamp, pv

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    try:
        ts, pv = forecast_one_step()
        print(f"\n===== PV NOW FINAL INDUSTRIAL =====")
        print(f"{ts} → {pv:.3f} kW")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR : {e}")