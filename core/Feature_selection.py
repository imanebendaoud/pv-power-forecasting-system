from sklearn.ensemble import RandomForestRegressor
import numpy as np

#charger les modes les plus importants 

def rf_per_vmd_modes(train, threshold=0.95):
    top_modes = list(np.load('top_modes.npy', allow_pickle=True))
    top_modes.append("res")
    feature_cols = [
        "ambient_temperature","wind_speed","module_temperature","tilted_radiation","global_radiation","humidity",
        "hour_sin","temp_radiation","hour_cos","is_day","radiation_accel","pv_lag","energy_roll_std","radiation_diff"]
    #,"entropy_mode_0","entropy_mode_1","entropy_mode_2","entropy_mode_3","entropy_mode_4","entropy_mode_5"
        
    selected_features = {}
    for i in top_modes:
        if i == "res":
           mode_col = "mode_res"
        else:
          mode_col = f"mode_{i}"
        X = train[feature_cols]
        y = train[mode_col]
        rf = RandomForestRegressor(n_estimators=200, random_state=42,n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        selected = []
        cumulative = 0
        for f, imp in ranked:
            cumulative += imp
            selected.append(f)
            if cumulative >= threshold :#deux conditions pour choisir les meilleur features
                break
        print(f"\nMode {i} → Selected: {selected}")
        selected_features[mode_col] = selected
    return selected_features