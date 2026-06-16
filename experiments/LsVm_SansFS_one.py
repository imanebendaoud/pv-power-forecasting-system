import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(
        "C:/Users/hp/Desktop/Projet_fin_etude/core"
    )
)

from preprocessing_vmd import load_data_vmd
from fonction_sequence import create_sequence

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

# =====================================================
# LOAD DATA
# =====================================================

train, test, scaler_y = load_data_vmd()

# =====================================================
# FEATURES (NO FEATURE SELECTION)
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
        Dense(1)
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

    X_train, y_train = create_sequence(

        train,

        mode_col,

        features,

        WINDOW_SIZE
    )

    X_test, y_test = create_sequence(

        test,

        mode_col,

        features,

        WINDOW_SIZE
    )

    model = build_model(

        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.fit(

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

y_true = test["energy"].values[
    WINDOW_SIZE:
]

# =====================================================
# ALIGN
# =====================================================

min_len = min(
    len(y_true),
    len(pred_y)
)

y_true = y_true[:min_len]

pred_y = pred_y[:min_len]

# =====================================================
# INVERSE SCALE
# =====================================================

y_true_real = scaler_y.inverse_transform(
    y_true.reshape(-1,1)
)

pred_real = scaler_y.inverse_transform(
    pred_y.reshape(-1,1)
)

# =====================================================
# NIGHT FILTER
# =====================================================

radiation = test[
    "global_radiation"
].values[
    WINDOW_SIZE:
]

radiation = radiation[:min_len]

pred_real[
    radiation <= 0
] = 0

pred_real[
    pred_real < 0
] = 0

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

print("\n===== FINAL RESULTS =====")

print("MAE :", mae)

print("RMSE:", rmse)

print("R2  :", r2)
import matplotlib.pyplot as plt

n_points = 500

plt.figure(figsize=(12,5))

plt.plot(y_true_real[:n_points],
         label='Valeurs réelles')

plt.plot(pred_real[:n_points],
         label='Valeurs prédites')

plt.title('VMD-LSTM sans Feature Selection - Prévision One-Step')
plt.xlabel('Temps')
plt.ylabel('Puissance photovoltaïque (kW)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()