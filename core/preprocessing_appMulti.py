import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys,os
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))
from vmd_utils import compute_vmd

def load_and_preprocess():

    # =========================
    # 1. LOAD DATA
    # =========================
    data = pd.read_csv("data_ready.csv")

    # =========================
    # 2. CLEAN + TIME INDEX
    # =========================
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp")
    data.set_index("timestamp", inplace=True)
    data = data.resample("30min").mean()

    # fill missing
    data = data.interpolate()

    # sécurité 
    data = data.dropna()
    # =========================
    # 3. SPLIT ROBUST (IMPORTANT FIX)
    # =========================
    data = data.sort_index()
    train = data[data.index <= "2025-10-24 03:00:00"].copy()
    test  = data[data.index >  "2025-10-24 03:00:00"].copy()
    print("MIN date:", data.index.min())
    print("MAX date:", data.index.max())
    # =========================
    # 4. FEATURE ENGINEERING
    # =========================
    for df in [train, test]:

        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["radiation_diff"] = df["global_radiation"].diff().fillna(0)
        df["temp_radiation"] = df["ambient_temperature"] * df["global_radiation"]
        df["is_day"] = (df["global_radiation"] > 0).astype(int)
        df["pv_lag"] = df["energy"].shift(1).fillna(0)

    # =========================
    # 5. VMD (SAFE WINDOW FIX)
    # =========================
    WINDOW_VMD = 2000

    signal = train["energy"].values[-WINDOW_VMD:]

    modes = compute_vmd(signal, K=5)

# créer colonnes
    for i in range(5):
        train[f"mode_{i}"] = 0

# injecter uniquement sur la fin
    for i in range(modes.shape[1]):
        train.iloc[-WINDOW_VMD:, train.columns.get_loc(f"mode_{i}")] = modes[:, i]

# test (approximation)
    for i in range(5):
        test[f"mode_{i}"] = 0

    # =========================
    # 6. FEATURES LIST
    # =========================
    features = [
        "ambient_temperature",
        "wind_speed",
        "humidity",
        "global_radiation",
        "radiation_diff",
        "temp_radiation",
        "hour_sin",
        "hour_cos",
        "is_day",
        "pv_lag",
        "mode_0",
        "mode_1",
        "mode_2",
        "mode_3",
        "mode_4"
    ]

    target = "energy"

    # =========================
    # 7. SCALING (SAFE)
    # =========================
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train[features] = scaler_X.fit_transform(train[features])
    test[features]  = scaler_X.transform(test[features])

    train[target] = scaler_y.fit_transform(train[[target]])
    test[target]  = scaler_y.transform(test[[target]])

    # =========================
    # 8. FINAL CHECK (DEBUG SAFE)
    # =========================
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    if len(train) == 0:
        raise ValueError(" Train dataset vide → vérifier timestamps ou split")

    return train, test, scaler_X, scaler_y, features