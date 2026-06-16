import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

def load_and_preprocess():

    data = pd.read_csv("data_ready.csv")

    # =========================
    # TIME
    # =========================
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data.set_index("timestamp", inplace=True)

    # =========================
    # RESAMPLING
    # =========================
    data = data.resample("1h").mean()
    data.ffill(inplace=True)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    data["hour_float"] = data.index.hour + data.index.minute/60
    data["hour_sin"] = np.sin(2*np.pi*data["hour_float"]/24)
    data["hour_cos"] = np.cos(2*np.pi*data["hour_float"]/24)
    data.drop("hour_float", axis=1, inplace=True)

    # 🔥 PHYSIQUE
    data.loc[data["global_radiation"] == 0, "energy"] = 0

    # 🔥 lag
    data["pv_lag"] = data["energy"].shift(1)
    data["pv_lag"].bfill(inplace=True)

    # 🔥 dynamique
    data["radiation_diff"] = data["global_radiation"].diff().fillna(0)

    # 🔥 jour/nuit
    data["is_day"] = (data["global_radiation"] > 20).astype(int)
    data["temp_radiation"] = data["module_temperature"] * data["global_radiation"]

    data["energy_roll_std"] = data["energy"].rolling(6).std().fillna(0)

    data["radiation_accel"] = data["global_radiation"].diff().diff().fillna(0)
    # 🔥 nettoyage
    data["energy"] = data["energy"].clip(lower=0)

    # =========================
    # SPLIT
    # =========================
    train = data[data.index <= "2025-10-24 03:00:00"].copy()
    test  = data[data.index >  "2025-10-24 03:00:00"].copy()

    # =========================
    # SCALING
    # =========================
    standard_features = [
        "ambient_temperature",
        "wind_speed",
        "module_temperature",
        "tilted_radiation",
        "global_radiation",
        "pv_lag"
    ]

    scaler_features = StandardScaler()
    train[standard_features] = scaler_features.fit_transform(train[standard_features])
    test[standard_features]  = scaler_features.transform(test[standard_features])

    scaler_MinMax = MinMaxScaler()
    train[["humidity"]] = scaler_MinMax.fit_transform(train[["humidity"]])
    test[["humidity"]]  = scaler_MinMax.transform(test[["humidity"]])

    scaler_robust = RobustScaler()
    train[["radiation_diff"]] = scaler_robust.fit_transform(train[["radiation_diff"]])
    test[["radiation_diff"]]  = scaler_robust.transform(test[["radiation_diff"]])

    scaler_y = StandardScaler()
    train["energy"] = scaler_y.fit_transform(train[["energy"]])
    test["energy"]  = scaler_y.transform(test[["energy"]])

    # 🔥 sauvegardejoblib.dump(scaler_features, "scaler_features.save")
    joblib.dump(scaler_MinMax, "scaler_minmax.save")
    joblib.dump(scaler_robust, "scaler_robust.save")
    joblib.dump(scaler_y, "scaler_y.save")
    joblib.dump(scaler_features, "scaler_features.save")



    return train, test, scaler_y