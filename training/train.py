import os,sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocessing_vmd1 import load_data_vmd
from Feature_selection1 import rf_per_vmd_modes
from fonction_sequence import create_sequence

WINDOW_SIZE = 48

ARTIFACTS_DIR = "artifacts_app"
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ======================
# DATA
# ======================
train, test, scaler_y, scaler_std, scaler_minmax = load_data_vmd()

feature_selection = rf_per_vmd_modes(train)

top_modes = np.load("top_modes.npy", allow_pickle=True).tolist()
top_modes.append("res")

# ======================
# MODEL
# ======================
def build_model(input_shape):

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mae")
    return model

# ======================
# TRAIN LOOP
# ======================
for m in top_modes:

    mode_col = "mode_res" if m == "res" else f"mode_{m}"

    features = feature_selection[mode_col]
    features = [f for f in features if f in train.columns]

    print(f"\nTraining {mode_col} | features: {len(features)}")

    X_train, y_train = create_sequence(train, mode_col, features, WINDOW_SIZE)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=7, restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ],
        verbose=1
    )

    model.save(os.path.join(MODELS_DIR, f"model_{mode_col}.keras"))

# ======================
# SAVE ARTIFACTS
# ======================
joblib.dump(feature_selection, f"{ARTIFACTS_DIR}/feature_selection.pkl")
joblib.dump(scaler_y, f"{ARTIFACTS_DIR}/scaler_y.pkl")
joblib.dump(scaler_std, f"{ARTIFACTS_DIR}/scaler_std.pkl")
joblib.dump(scaler_minmax, f"{ARTIFACTS_DIR}/scaler_minmax.pkl")

print("TRAIN DONE ✔")