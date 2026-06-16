import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(
        "C:/Users/hp/Desktop/Projet_fin_etude/core"
    )
)

from preprocessing_vmd import load_data_vmd

from fonction_sequence import (
    create_sequence_multi
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# CONFIG
# =====================================================

WINDOW_SIZE = 48

HORIZON = 24

# =====================================================
# LOAD DATA
# =====================================================

train, test, scaler_y = load_data_vmd()

# =====================================================
# FEATURES
# =====================================================

features = [

    "ambient_temperature",

    "wind_speed",

    "module_temperature",

    "tilted_radiation",

    "global_radiation",

    "humidity",

    "pv_lag",

    "radiation_diff",

    "hour_sin",

    "hour_cos",

    "is_day",

    "temp_radiation",

    "energy_roll_std",

    "radiation_accel"
]

# =====================================================
# MODES
# =====================================================

top_modes = np.load(
    "top_modes.npy",
    allow_pickle=True
).tolist()

selected_modes = [
    f"mode_{m}"
    for m in top_modes
]

selected_modes.append("mode_res")

# =====================================================
# MODEL
# =====================================================

def build_model(input_shape):

    model = Sequential()

    model.add(

        LSTM(
            128,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(64)
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(HORIZON)
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# =====================================================
# TRAIN
# =====================================================

predictions = []

for mode_col in selected_modes:

    print(f"\n🔥 TRAINING {mode_col}")

    X_train, y_train = create_sequence_multi(

        train,

        mode_col,

        features,

        WINDOW_SIZE,

        HORIZON
    )

    X_test, y_test = create_sequence_multi(

        test,

        mode_col,

        features,

        WINDOW_SIZE,

        HORIZON
    )

    model = build_model(

        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    history=model.fit(

        X_train,

        y_train,

        epochs=40,

        batch_size=32,

        validation_split=0.2,

        callbacks=[
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ],

        verbose=1
    )

    pred = model.predict(X_test)

    predictions.append(pred)

# =====================================================
# RECONSTRUCTION
# =====================================================

pred_y = np.sum(
    predictions,
    axis=0
)

# =====================================================
# TRUE VALUES
# =====================================================

y_true = []

energy = test["energy"].values

for i in range(

    len(test)

    - WINDOW_SIZE

    - HORIZON
):

    y_true.append(

        energy[
            i+WINDOW_SIZE:
            i+WINDOW_SIZE+HORIZON
        ]
    )

y_true = np.array(y_true)

# =====================================================
# RESHAPE
# =====================================================

y_true_flat = y_true.reshape(-1, 1)
pred_flat = pred_y.reshape(-1, 1)

# =====================================================
# INVERSE SCALE
# =====================================================

y_true_real = scaler_y.inverse_transform(y_true_flat)
pred_real = scaler_y.inverse_transform(pred_flat)

# =====================================================
# NIGHT FILTER
# =====================================================

radiation = []

global_rad = test["global_radiation"].values

for i in range(len(test) - WINDOW_SIZE - HORIZON):

    radiation.extend(
        global_rad[
            i + WINDOW_SIZE :
            i + WINDOW_SIZE + HORIZON
        ]
    )

radiation = np.array(radiation)

pred_real[radiation <= 0] = 0
pred_real[pred_real < 0] = 0

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    y_true_real,
    pred_real
)

rmse = np.sqrt(
    mean_squared_error(
        y_true_real,
        pred_real
    )
)

r2 = r2_score(
    y_true_real,
    pred_real
)

print("\n===== MULTISTEP RESULTS =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =====================================================
# RECONSTRUCTION FORMAT 24 HORIZONS
# =====================================================

y_true_real_24 = y_true_real.reshape(y_true.shape)

pred_real_24 = pred_real.reshape(pred_y.shape)

print("Shape réel :", y_true_real_24.shape)
print("Shape prédit :", pred_real_24.shape)

# =====================================================
# GRAPHIQUE 24H
# =====================================================

sample = 2

plt.figure(figsize=(10,5))

plt.plot(
    y_true_real_24[sample],
    marker="o",
    linewidth=2,
    label="Réel"
)

plt.plot(
    pred_real_24[sample],
    marker="s",
    linewidth=2,
    label="Prédit"
)

plt.xlabel("Horizon de prévision")
plt.ylabel("Puissance PV (kW)")
plt.title("Prévision Multi-Step (24 horizons)")
plt.grid(True)
plt.legend()

plt.show()

sample = 50

plt.figure(figsize=(12,5))

plt.plot(
    y_true_real_24[sample],
    linewidth=3,
    marker="o",
    label="Valeurs réelles"
)

plt.plot(
    pred_real_24[sample],
    linewidth=3,
    marker="s",
    label="Valeurs prédites"
)

plt.xlabel("Pas de prévision")
plt.ylabel("Puissance PV (kW)")
plt.title("Comparaison des valeurs réelles et prédites sur un horizon de 24 pas")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.title("LSTM classique - Évolution de l'apprentissage")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()