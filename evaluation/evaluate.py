import os, sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing_vmd1 import load_data_vmd
from fonction_sequence import create_sequence_multi

# =====================
# LOAD DATA
# =====================
train, test, scaler_y, scaler_std, scaler_minmax = load_data_vmd()

WINDOW = 48
HORIZON = 24

# =====================
# LOAD ARTIFACTS
# =====================
feature_selection = joblib.load("artifacts/feature_selection.pkl")

top_modes = np.load("top_modes.npy", allow_pickle=True).tolist()
top_modes.append("res")

models = {}

for m in top_modes:
    mode_col = f"mode_{m}" if m != "res" else "mode_res"
    models[m] = load_model(f"artifacts/models/model_{mode_col}.keras")

# =====================
# EVALUATION CLEAN
# =====================
preds = []

for m in top_modes:

    mode_col = f"mode_{m}" if m != "res" else "mode_res"

    features = feature_selection[mode_col]
    features = [f for f in features if f in test.columns]

    X_test, y_test = create_sequence_multi(
        test, mode_col, features, WINDOW, HORIZON
    )

    y_pred = models[m].predict(X_test, verbose=0)

    preds.append(y_pred)

# =====================
# RECONSTRUCTION CORRECTE
# =====================
y_pred = np.sum(preds, axis=0)

y_true = []
y_test_energy = test["energy"].values

for i in range(len(test) - WINDOW - HORIZON):
    y_true.append(
        y_test_energy[i + WINDOW : i + WINDOW + HORIZON]
    )

y_true = np.array(y_true)

# ALIGNMENT
min_len = min(len(y_true), len(y_pred))

y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

# =====================
# INVERSE SCALING
# =====================
y_true = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()

# =====================
# METRICS
# =====================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n===== FINAL EVALUATION =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)