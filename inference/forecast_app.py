import os
import numpy as np
import pandas as pd
import joblib
import requests
import pvlib
from tensorflow.keras.models import load_model

# =====================================================
# PATHS
# =====================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS  = os.path.join(BASE_DIR, "..", "artifacts_app")
MODELS_DIR = os.path.join(ARTIFACTS, "models")

# =====================================================
# LOAD ARTIFACTS
# =====================================================
top_modes = np.load(
    os.path.join(BASE_DIR, "..", "top_modes.npy"),
    allow_pickle=True
).tolist()
top_modes.append("res")

feature_selection = joblib.load(os.path.join(ARTIFACTS, "feature_selection.pkl"))
scaler_y          = joblib.load(os.path.join(ARTIFACTS, "scaler_y.pkl"))
scaler_std        = joblib.load(os.path.join(ARTIFACTS, "scaler_std.pkl"))
scaler_minmax     = joblib.load(os.path.join(ARTIFACTS, "scaler_minmax.pkl"))

# max physique observé dans les données d'entraînement
PV_MAX = 57.94

# =====================================================
# LOAD MODELS
# =====================================================
models = {}
for m in top_modes:
    mode_col  = "mode_res" if m == "res" else f"mode_{m}"
    models[m] = load_model(os.path.join(MODELS_DIR, f"model_{mode_col}.keras"))

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
# TIMESERIES FIX
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
    df["clearness_index"] = np.clip(
        df["global_radiation"] / (df["ghi"] + 1e-6), 0, 1.2
    )
    df["is_day"] = (df["solar_elevation"] > 0).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate().ffill().fillna(0)
    return df

# =====================================================
# SCALING
# =====================================================
def scale_features(df):
    is_day_raw  = df["is_day"].values.copy()
    std_cols    = scaler_std.feature_names_in_
    minmax_cols = scaler_minmax.feature_names_in_
    df[std_cols]    = scaler_std.transform(df.reindex(columns=std_cols,    fill_value=0))
    df[minmax_cols] = scaler_minmax.transform(df.reindex(columns=minmax_cols, fill_value=0))
    df["is_day"]    = is_day_raw
    return df

# =====================================================
# CORE PREDICTION
# =====================================================
WINDOW = 48
def _predict_at(df_scaled, now_idx, elev_raw, rad_raw, debug=False):

    start_idx = max(0, now_idx - WINDOW + 1)
    df_window = df_scaled.iloc[start_idx: now_idx + 1].copy()

    if len(df_window) < WINDOW:
        pad = pd.DataFrame(
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

        X = df_window[features].values.reshape(
            1,
            WINDOW,
            len(features)
        )

        pred = models[m](X, training=False).numpy()

        if debug:
            #pv_mode = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
            print( f"mode {m:>3} -> scaled={pred[0][0]:.4f}")
            
        predictions.append(pred)

    if not predictions:
        raise ValueError("Aucune prédiction produite")

    final = np.sum(predictions, axis=0)

    if debug:
        print("\nDEBUG SCALÉ")
        print("final scaled =", final.flatten()[0])
        print("scaler mean  =", scaler_y.mean_[0])
        print("scaler std   =", scaler_y.scale_[0])

    pv = scaler_y.inverse_transform(
        final.reshape(-1, 1)
    ).flatten()[0]

    pv = float(pv)

    # limite physique de la centrale
    pv = min(pv, PV_MAX)

    # nuit
    if elev_raw < 3 or rad_raw < 20:
        pv = 0.0
    # lever / coucher
    elif elev_raw < 10:

        elev_factor = elev_raw / 10.0
        rad_factor = min(rad_raw / 300.0, 1.0)

        correction = min(
            elev_factor,
            rad_factor
        )

        pv *= correction

        if debug:
            print("\nCORRECTION PHYSIQUE")
            print("elev_factor =", elev_factor)
            print("rad_factor  =", rad_factor)
            print("correction  =", correction)

    pv = max(pv, 0.0)

    if debug:
        print(f"élévation réelle : {elev_raw:.2f}°")
        print(f"radiation réelle : {rad_raw:.1f} W/m²")
        print(f"pv final         : {pv:.4f} kW")

    return pv

# =====================================================
# FORECAST TEMPS RÉEL
# =====================================================
def forecast_one_step(debug=False):
    now = pd.Timestamp.now(tz="Africa/Casablanca")

    df  = get_weather()
    df  = fix_timeseries(df)
    df  = preprocess(df)

    now_idx = df.index.get_indexer([now], method="pad")[0]

    # lire élévation et radiation sur df NON scalé
    elev_raw = float(df["solar_elevation"].iloc[now_idx])
    rad_raw  = float(df["global_radiation"].iloc[now_idx])

    if debug:
        print(f"\n--- Debug forecast ---")
        print(f"Maintenant       : {now}")
        print(f"Timestamp nearest: {df.index[now_idx]}")
        print(f"solar_elevation  : {elev_raw:.2f}°")
        print(f"global_radiation : {rad_raw:.1f} W/m²")
        print(f"is_day           : {int(df['is_day'].iloc[now_idx])}")
        print(f"Fenêtre : {df.index[max(0, now_idx-WINDOW+1)]}  →  {df.index[now_idx]}")

    df_scaled = scale_features(df.copy())
    pv = _predict_at(
    df_scaled,
    now_idx,
    elev_raw,
    rad_raw,
    debug=debug)

    if debug:
        print(f"pv final  : {pv:.4f} kW")

    ts = now + pd.Timedelta(minutes=30)
    print("\nDEBUG PHYSIQUE")
    print("timestamp =", df.index[now_idx])
    print("elevation =", elev_raw)
    print("ghi       =", rad_raw)
    print("tilted    =", float(df["tilted_radiation"].iloc[now_idx]))
    print("pv final  =", pv)
    return ts, pv

# =====================================================
# TEST SUR TIMESTAMP PRÉCIS
# =====================================================
def forecast_at(target_time_str, debug=False):
    target  = pd.Timestamp(target_time_str, tz="Africa/Casablanca")
    df      = get_weather()
    df      = fix_timeseries(df)
    df      = preprocess(df)

    now_idx  = df.index.get_indexer([target], method="nearest")[0]
    elev_raw = float(df["solar_elevation"].iloc[now_idx])
    rad_raw  = float(df["global_radiation"].iloc[now_idx])

    if debug:
        print(f"\n--- Test à {df.index[now_idx]} ---")
        print(f"solar_elevation  : {elev_raw:.2f}°")
        print(f"global_radiation : {rad_raw:.1f} W/m²")

    df_scaled = scale_features(df.copy())

    pv = _predict_at(
    df_scaled,
    now_idx,
    elev_raw,
    rad_raw,
    debug=debug
)

    if debug:
        print(f"pv final : {pv:.4f} kW")
    return pv

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    ts, pv = forecast_one_step(debug=True)
    forecast_at("2026-06-07 08:00:00", debug=True)
    forecast_at("2026-06-07 12:00:00", debug=True)
    forecast_at("2026-06-07 18:30:00", debug=True)
    forecast_at("2026-06-07 20:00:00", debug=True)
    print(f"\n{ts} → {pv:.3f} kW")
