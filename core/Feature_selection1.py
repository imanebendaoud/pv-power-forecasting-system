from sklearn.ensemble import RandomForestRegressor
import numpy as np

def rf_per_vmd_modes(train, threshold=0.95):

    top_modes = list(
        np.load("top_modes.npy", allow_pickle=True)
    )

    top_modes.append("res")

    # =========================================
    # FEATURES DISPONIBLES
    # =========================================

    feature_cols = [

        # météo
        "ambient_temperature",
        "wind_speed",
        "humidity",
        "clouds",

        # radiation
        "global_radiation",
        "tilted_radiation",
        "ghi",
        "dni",
        "dhi",

        # solaire
        "zenith",
        "solar_elevation",

        # temporelles
        "hour_sin",
        "hour_cos",
        "is_day",

        # engineered
        "temp_radiation",]

    # garder seulement colonnes existantes
    feature_cols = [
        f for f in feature_cols
        if f in train.columns
    ]

    selected_features = {}

    # =========================================
    # FEATURE SELECTION
    # =========================================

    for i in top_modes:

        if i == "res":
            mode_col = "mode_res"
        else:
            mode_col = f"mode_{i}"

        if mode_col not in train.columns:
            continue

        X = train[feature_cols]

        y = train[mode_col]

        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X, y)

        importances = rf.feature_importances_

        ranked = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []

        cumulative = 0

        for f, imp in ranked:

            cumulative += imp

            selected.append(f)

            if cumulative >= threshold:
                break

        print(f"\nMode {i}")
        print("Selected features :", selected)

        selected_features[mode_col] = selected

    return selected_features
