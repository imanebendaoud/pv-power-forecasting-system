# models/train_model.py
import sys, os
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing_vmd import load_data_vmd
from Feature_selection import rf_per_vmd_modes
from fonction_sequence import create_sequence_multi

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pickle
import os

def weighted_mse(y_true, y_pred):
    weight = 1 + 5 * tf.abs(y_true)
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))

def create_lstm_model(input_shape, horizon=24):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(horizon))
    model.compile(optimizer="adam", loss="mse")
    return model

# 1. Charger les données
train, test, scaler_y = load_data_vmd()

top_modes = np.load('top_modes.npy', allow_pickle=True).tolist()
top_modes.append("res")

feature_selection = rf_per_vmd_modes(train)

# 2. Sauvegarder top_modes et feature_selection
resulats_dir = "C:/Users/hp/Desktop/Projet_fin_etude/resulats"
os.makedirs(resulats_dir, exist_ok=True)

with open(os.path.join(resulats_dir, "top_modes.pkl"), "wb") as f:
    pickle.dump(top_modes, f)

with open(os.path.join(resulats_dir, "feature_selection.pkl"), "wb") as f:
    pickle.dump(feature_selection, f)

# 3. Entraîner et sauvegarder les modèles
models = []

for i, mode_name in enumerate(top_modes):
    if mode_name == "res":
        mode_col = "mode_res"
    else:
        mode_col = f"mode_{mode_name}"

    features = feature_selection[mode_col]

    x_train, y_train = create_sequence_multi(
        train, mode_col, features, window_size=48, horizon=24
    )
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = create_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"Entraînement du modèle pour {mode_col}...")
    model.fit(
        x_train, y_train,
        epochs=60,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Sauvegarder le modèle (format .keras)
    model_path = os.path.join(resulats_dir, f"model_mode_{i}.keras")
    model.save(model_path)

    models.append(model)