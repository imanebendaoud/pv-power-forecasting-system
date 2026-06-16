import os
import sys

sys.path.append(
    os.path.abspath(
        "C:/Users/hp/Desktop/Projet_fin_etude/core"
    )
)

import numpy as np
import joblib

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing_vmd1 import load_data_vmd
from fonction_sequence import create_sequence

# =====================================================
# LOAD DATA
# =====================================================

train, test, scaler_y, scaler_std, scaler_minmax = load_data_vmd()

# =====================================================
# CONFIG
# =====================================================

WINDOW = 48

# =====================================================
# LOAD ARTIFACTS
# =====================================================

ARTIFACTS = "artifacts_app"

MODELS_DIR = os.path.join(
    ARTIFACTS,
    "models"
)

feature_selection = joblib.load(
    os.path.join(
        ARTIFACTS,
        "feature_selection.pkl"
    )
)

top_modes = np.load(
    "top_modes.npy",
    allow_pickle=True
).tolist()

top_modes.append("res")

# =====================================================
# LOAD MODELS
# =====================================================

models = {}

for m in top_modes:

    mode_col = (
        f"mode_{m}"
        if m != "res"
        else "mode_res"
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"model_{mode_col}.keras"
    )

    models[m] = load_model(model_path)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = []

for m in top_modes:

    mode_col = (
        f"mode_{m}"
        if m != "res"
        else "mode_res"
    )

    features = feature_selection[
        mode_col
    ]

    features = [

        f for f in features

        if f in test.columns
    ]

    # =========================================
    # CREATE SEQUENCES
    # =========================================

    X_test, y_test = create_sequence(

        test,

        mode_col,

        features,

        WINDOW
    )

    # =========================================
    # MODEL PREDICTION
    # =========================================

    y_pred = models[m].predict(
        X_test,
        verbose=0
    )

    predictions.append(y_pred)

# =====================================================
# FINAL RECONSTRUCTION
# =====================================================

y_pred = np.sum(
    predictions,
    axis=0
)

# =====================================================
# TRUE ENERGY
# =====================================================

y_true = test["energy"].values[
    WINDOW:
]

# ALIGNMENT

min_len = min(
    len(y_true),
    len(y_pred)
)

y_true = y_true[:min_len]

y_pred = y_pred[:min_len]

# =====================================================
# INVERSE SCALING
# =====================================================

y_true = scaler_y.inverse_transform(
    y_true.reshape(-1,1)
).flatten()

y_pred = scaler_y.inverse_transform(
    y_pred.reshape(-1,1)
).flatten()
# =====================================================
# PHYSICAL CONSTRAINT
# =====================================================

solar_elevation = test[
    "solar_elevation"
].values[WINDOW:]

solar_elevation = solar_elevation[:len(y_pred)]

y_pred[solar_elevation <= 0] = 0
# =====================================================
# CLEAN
# =====================================================

y_pred = np.nan_to_num(
    y_pred,
    nan=0
)

y_pred = np.clip(
    y_pred,
    0,
    None
)

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

r2 = r2_score(
    y_true,
    y_pred
)

# =====================================================
# RESULTS
# =====================================================

print("\n===== ONE STEP EVALUATION =====")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R2   : {r2:.4f}")

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

print("\n===== SAMPLE PREDICTIONS =====")

for i in range(10):

    print(
        f"REAL={y_true[i]:.2f} kW | "
        f"PRED={y_pred[i]:.2f} kW"
    )