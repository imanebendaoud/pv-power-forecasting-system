import os
import sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing import load_and_preprocess
from fonction_sequence import create_sequence

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

X_train, y_train = create_sequence(
    train,
    "energy",
    feature_cols=features,
    window_size=48
)

X_test, y_test = create_sequence(
    test,
    "energy",
    feature_cols=features,
    window_size=48
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

model.add(Dense(1))

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

model.fit(
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
)

y_pred_inv = scaler_y.inverse_transform(
    prediction
)

y_pred_inv[y_pred_inv < 0] = 0

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(y_test_inv, y_pred_inv)

rmse = np.sqrt(
    mean_squared_error(y_test_inv, y_pred_inv)
)

r2 = r2_score(y_test_inv, y_pred_inv)

mape = np.mean(
    np.abs(
        (y_test_inv - y_pred_inv)
        /(y_test_inv + 1e-6)
    )
)*100

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("MAPE:", mape)

import matplotlib.pyplot as plt

n_points = 500   # nombre de points à afficher

plt.figure(figsize=(12,5))

plt.plot(y_test_inv[:n_points],
         label='Valeurs réelles')

plt.plot(y_pred_inv[:n_points],
         label='Valeurs prédites')

plt.title('LSTM-GRU - Prévision One-Step')
plt.xlabel('Temps')
plt.ylabel('Puissance photovoltaïque (kW)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()