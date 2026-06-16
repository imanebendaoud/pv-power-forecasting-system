import pandas as pd
import numpy as np
import pvlib

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from vmdpy import VMD

# =========================
# CONFIG
# =========================

VMD_K = 12
RESAMPLE_RULE = "30min"

# =========================
# VMD
# =========================

def apply_vmd(signal, K=VMD_K):

    u, _, _ = VMD(
        signal,
        alpha=1700,
        tau=0,
        K=K,
        DC=0,
        init=1,
        tol=1e-7
    )

    return u

# =========================
# MAIN
# =========================
def get_feature_columns():
    return [
        "ambient_temperature",
        "wind_speed",
        "module_temperature",
        "ghi",
        "dni",
        "dhi",
        "global_radiation",
        "tilted_radiation",
        "temp_radiation",
        "zenith",
        "solar_elevation",
        "clearness_index",
        "hour_sin",
        "hour_cos",
        "humidity",
        "clouds",
        "is_day"
    ]
def load_data_vmd(path="data_ready.csv"):

    # =========================
    # LOAD
    # =========================

    data = pd.read_csv(path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = data.set_index("timestamp")
    data = data.sort_index()
    data = data[~data.index.duplicated(keep='first')]
    # =========================
    # RESAMPLE
    # =========================

    data = data.resample(
        RESAMPLE_RULE
    ).mean()

    data = data.interpolate()

    data = data.ffill()

    # =========================
    # LOCATION
    # =========================

    location = pvlib.location.Location(

        latitude=30.92,
        longitude=-6.91,
        tz="Africa/Casablanca"
    )

    # =========================
    # SOLAR POSITION
    # =========================

    solpos = location.get_solarposition(
        data.index
    )

    data["zenith"] = solpos["zenith"]

    data["solar_elevation"] = (
        solpos["elevation"]
    )

    # =========================
    # CLEAR SKY
    # =========================

    cs = location.get_clearsky(
        data.index
    )

    data["ghi"] = cs["ghi"]

    data["dni"] = cs["dni"]

    data["dhi"] = cs["dhi"]

    # =========================
    # CLOUDS
    # =========================

    if "clouds" not in data.columns:
        data["clouds"] = 0

    data["clouds"] = np.clip(
        data["clouds"],
        0,
        100
    )

    # =========================
    # REMOVE NIGHT NOISE
    # =========================

    data.loc[
        data["solar_elevation"] <= 0,
        "global_radiation"
    ] = 0

    # =========================
    # TILTED RADIATION
    # =========================

    poa = pvlib.irradiance.get_total_irradiance(

        surface_tilt=25,

        surface_azimuth=180,

        dni=data["dni"],

        ghi=data["global_radiation"],

        dhi=data["dhi"],

        solar_zenith=data["zenith"],

        solar_azimuth=solpos["azimuth"]
    )

    data["tilted_radiation"] = (
        poa["poa_global"]
    )

    # =========================
    # MODULE TEMP
    # =========================

    data["module_temperature"] = (

        data["ambient_temperature"]

        + 0.03 * data["tilted_radiation"]
    )

    # =========================
    # CYCLICAL FEATURES
    # =========================

    hour_float = (

        data.index.hour

        + data.index.minute / 60
    )

    data["hour_sin"] = np.sin(
        2 * np.pi * hour_float / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * hour_float / 24
    )

    # =========================
    # EXTRA FEATURES
    # =========================

    data["temp_radiation"] = (

        data["module_temperature"]

        * data["tilted_radiation"]
    )

    data["clearness_index"] = (

        data["global_radiation"]

        / (data["ghi"] + 1e-6)
    )

    data["is_day"] = (
        data["solar_elevation"] > 0
    ).astype(int)

    # =========================
    # CLEAN
    # =========================

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    data = data.interpolate()

    data = data.ffill()

    data = data.fillna(0)

    # =========================
    # SPLIT
    # =========================

    split_date = "2025-10-24 03:00:00"

    train = data[
        data.index <= split_date
    ].copy()

    test = data[
        data.index > split_date
    ].copy()

    # =========================
    # SCALING
    # =========================

    std_features = [

        "ambient_temperature",

        "wind_speed",

        "module_temperature",

        "ghi",

        "dni",

        "dhi",

        "global_radiation",

        "tilted_radiation",

        "temp_radiation",

        "zenith",

        "solar_elevation",

        "clearness_index"
    ]

    minmax_features = [
        "humidity",
        "clouds"
    ]

    scaler_std = StandardScaler()

    scaler_minmax = MinMaxScaler()

    train[std_features] = (
        scaler_std.fit_transform(
            train[std_features]
        )
    )

    test[std_features] = (
        scaler_std.transform(
            test[std_features]
        )
    )

    train[minmax_features] = (
        scaler_minmax.fit_transform(
            train[minmax_features]
        )
    )

    test[minmax_features] = (
        scaler_minmax.transform(
            test[minmax_features]
        )
    )

    # =========================
    # TARGET
    # =========================

    scaler_y = StandardScaler()

    train["energy"] = (
        scaler_y.fit_transform(
            train[["energy"]]
        )
    )

    test["energy"] = (
        scaler_y.transform(
            test[["energy"]]
        )
    )

    # =========================
    # VMD
    # =========================

    full_signal = np.concatenate([

        train["energy"].values,

        test["energy"].values
    ])

    full_modes = apply_vmd(
        full_signal,
        K=VMD_K
    )

    split_idx = len(train)

    train_modes = full_modes[:, :split_idx]

    test_modes = full_modes[:, split_idx:]

    # =========================
    # TOP MODES
    # =========================

    energy_ratio = []

    for i in range(train_modes.shape[0]):

        energy_ratio.append(
            np.sum(train_modes[i]**2)
        )

    top_modes = np.argsort(
        energy_ratio
    )[-6:]

    np.save(
        "top_modes.npy",
        top_modes
    )

    # =========================
    # ALIGNMENT
    # =========================

    min_train = min(
        len(train),
        train_modes.shape[1]
    )

    min_test = min(
        len(test),
        test_modes.shape[1]
    )

    train = train.iloc[:min_train].copy()

    test = test.iloc[:min_test].copy()

    train_modes = train_modes[:, :min_train]

    test_modes = test_modes[:, :min_test]

    # =========================
    # RESIDUAL
    # =========================

    res_train = (

        train["energy"].values

        - np.sum(train_modes, axis=0)
    )

    res_test = (

        test["energy"].values

        - np.sum(test_modes, axis=0)
    )

    train["mode_res"] = res_train

    test["mode_res"] = res_test

    # =========================
    # ADD MODES
    # =========================

    for i in top_modes:

        train[f"mode_{i}"] = (
            train_modes[i]
        )

        test[f"mode_{i}"] = (
            test_modes[i]
        )

    return (
        train,
        test,
        scaler_y,
        scaler_std,
        scaler_minmax
    )
x,y,c,v,b=load_data_vmd()