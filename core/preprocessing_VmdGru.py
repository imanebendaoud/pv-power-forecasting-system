import numpy as np
from preprocessing_vmd import load_data_vmd
from Feature_selection import rf_per_vmd_modes

# ----------------------------
# Sélection des meilleurs modes
# ----------------------------
def select_modes_by_correlation(train, num_modes=4):
    mode_cols = [col for col in train.columns if "mode_" in col]

    correlations = {}
    for col in mode_cols:
        correlations[col] = abs(train[col].corr(train["energy"]))

    sorted_modes = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    selected_modes = [mode for mode, _ in sorted_modes[:num_modes]]

    print("Selected modes:", selected_modes)

    return selected_modes


# ----------------------------
# Création des séquences multi-step
# ----------------------------
def create_sequence_multi(data, target, feature_cols, window_size=48, horizon=24):

    X, y = [], []
    values = data[feature_cols + [target]].values
    target_index = len(feature_cols)

    for i in range(len(data) - window_size - horizon):
        X.append(values[i:i+window_size, :-1])
        y.append(values[i+window_size:i+window_size+horizon, target_index])

    return np.array(X), np.array(y)


# ----------------------------
# Fonction principale
# ----------------------------
def prepare_data(window_size=24, horizon=24):

    train, test, scaler_y = load_data_vmd()
    feature_selection = rf_per_vmd_modes(train)

    # 🔥 garder seulement les meilleurs modes
    selected_modes = ["mode_5", "mode_4", "mode_res"]

    data_dict = {}

    for mode_col in selected_modes:
        features = feature_selection[mode_col]

        X_train, y_train = create_sequence_multi(
            train, mode_col, features, window_size, horizon
        )

        X_test, y_test = create_sequence_multi(
            test, mode_col, features, window_size, horizon
        )

        data_dict[mode_col] = (X_train, y_train, X_test, y_test)

    return data_dict, selected_modes, test