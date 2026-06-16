import os
import sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing import load_and_preprocess
from fonction_sequence import create_sequence_multi

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# CONFIG
# =====================================

WINDOW_SIZE = 48
HORIZON = 24

# =====================================
# LOAD DATA
# =====================================

train, test, scaler_y = load_and_preprocess()

# =====================================
# FEATURES
# =====================================

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

# =====================================
# SEQUENCES
# =====================================

X_train, y_train = create_sequence_multi(
    train,
    "energy",
    features,
    WINDOW_SIZE,
    HORIZON
)

X_test, y_test = create_sequence_multi(
    test,
    "energy",
    features,
    WINDOW_SIZE,
    HORIZON
)

# =====================================
# MODEL
# =====================================

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(
    GRU(64)
)

model.add(Dropout(0.2))

model.add(Dense(HORIZON))

model.compile(
    optimizer="adam",
    loss="mse"
)

# =====================================
# TRAIN
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history=model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# =====================================
# PREDICTION
# =====================================

prediction = model.predict(X_test)

# =====================================
# INVERSE SCALE
# =====================================

y_test_inv = scaler_y.inverse_transform(
    y_test.reshape(-1,1)
).reshape(y_test.shape)

y_pred_inv = scaler_y.inverse_transform(
    prediction.reshape(-1,1)
).reshape(prediction.shape)

y_pred_inv[y_pred_inv < 0] = 0

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(
    y_test_inv.flatten(),
    y_pred_inv.flatten()
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_inv.flatten(),
        y_pred_inv.flatten()
    )
)

r2 = r2_score(
    y_test_inv.flatten(),
    y_pred_inv.flatten()
)

mape = np.mean(
    np.abs(
        (y_test_inv.flatten() - y_pred_inv.flatten())
        /(y_test_inv.flatten() + 1e-6)
    )
)*100

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("MAPE:", mape)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
idx = 0

plt.figure(figsize=(10,5))

plt.plot(
    y_test_inv[idx],
    marker='o',
    linewidth=2,
    label='Réel'
)

plt.plot(
    y_pred_inv[idx],
    marker='s',
    linewidth=2,
    label='Prédit'
)

plt.title(
    "Modèle proposé (VMD-LSTM + FS)\nPrévision Multi-Step sur 24 horizons"
)

plt.xlabel("Horizon de prévision")
plt.ylabel("Puissance PV (kW)")
plt.legend()
plt.grid(True)

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

 
plt.figure(figsize=(10,5))

plt.plot(np.mean(y_test_inv[:500], axis=1), label="Réel (moyenne 24h)")
plt.plot(np.mean(y_pred_inv[:500], axis=1), label="Prédit (moyenne 24h)")

plt.title("Multi-Step - Vision globale")
plt.legend()
plt.grid()
plt.show()