import os,sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

import numpy as np
import joblib
import tensorflow as tf

from preprocessing_vmd1 import load_data_vmd
from Feature_selection1 import rf_per_vmd_modes
from fonction_sequence import create_sequence_multi
from model_Multi import model_LSTM_multi

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =====================
# FIX RANDOM SEED
# =====================

np.random.seed(42)
tf.random.set_seed(42)

# =====================
# LOAD DATA
# =====================

train, test, scaler_y, scaler_std, scaler_minmax = load_data_vmd()

# =====================
# FEATURE SELECTION
# =====================

feature_selection = rf_per_vmd_modes(train.fillna(0))

os.makedirs("artifacts/models", exist_ok=True)

joblib.dump(feature_selection, "artifacts/feature_selection.pkl")
joblib.dump(scaler_y, "artifacts/scaler_y.pkl")
joblib.dump(scaler_std, "artifacts/scaler_std.pkl")
joblib.dump(scaler_minmax, "artifacts/scaler_minmax.pkl")

# =====================
# MODES
# =====================

top_modes = np.load(
    "top_modes.npy",
    allow_pickle=True
).tolist()

top_modes.append("res")

WINDOW = 48
HORIZON = 24

# =====================
# TRAIN LOOP
# =====================

for m in top_modes:

    mode_col = (
        f"mode_{m}"
        if m != "res"
        else "mode_res"
    )

    print(f"\nTRAINING {mode_col}")

    features = feature_selection[mode_col]
 
    features = [
        f for f in features
        if f in train.columns
    ]

    X_train, y_train = create_sequence_multi(
        train,
        mode_col,
        features,
        WINDOW,
        HORIZON
    )

    model = model_LSTM_multi(
        (WINDOW, len(features))
    )

    # =====================
    # CALLBACKS
    # =====================

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5
    )

    checkpoint = ModelCheckpoint(
        filepath=f"artifacts/models/model_{mode_col}.keras",
        monitor="val_loss",
        save_best_only=True
    )

    # =====================
    # TRAIN
    # =====================

    history = model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            early_stop,
            reduce_lr,
            checkpoint
        ],
        verbose=1
    )

print("\nTRAINING DONE")