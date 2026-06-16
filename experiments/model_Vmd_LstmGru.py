import os
import sys
import numpy as np
import joblib

sys.path.append(
    os.path.abspath(
        "C:/Users/hp/Desktop/Projet_fin_etude/core"
    )
)

from preprocessing_vmd import load_data_vmd
from Feature_selection import rf_per_vmd_modes

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
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

feature_selection = rf_per_vmd_modes(train)

# =====================================================
# SELECT MODES
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

print("\nSELECTED MODES:")
print(selected_modes)

# =====================================================
# CREATE SEQUENCE (ONE STEP)
# =====================================================

def create_sequence(
    data,
    target,
    feature_cols,
    window_size=48
):

    X = []
    y = []

    values = data[
        feature_cols + [target]
    ].values

    target_index = len(feature_cols)

    for i in range(
        len(data) - window_size
    ):

        X.append(
            values[
                i:i+window_size,
                :-1
            ]
        )

        y.append(
            values[
                i+window_size,
                target_index
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )

# =====================================================
# MODEL
# =====================================================

def build_lstm_gru_model(input_shape):

    model = Sequential()

    model.add(

        LSTM(

            128,

            return_sequences=True,

            input_shape=input_shape
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dropout(0.2)
    )

    model.add(

        GRU(64)
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(

        optimizer="adam",

        loss="mae"
    )

    return model

# =====================================================
# CALLBACKS
# =====================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    verbose=1
)

# =====================================================
# TRAIN
# =====================================================

predictions = []

for mode_col in selected_modes:

    print("\n===================================")
    print(f"TRAINING {mode_col}")
    print("===================================")

    # =========================================
    # FEATURES
    # =========================================

    features = feature_selection[
        mode_col
    ]

    features = [

        f for f in features

        if f in train.columns
    ]

    print("NB FEATURES:", len(features))

    # =========================================
    # SEQUENCES
    # =========================================

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

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # =========================================
    # MODEL
    # =========================================

    model = build_lstm_gru_model(

        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    # =========================================
    # TRAIN
    # =========================================

    model.fit(

        X_train,

        y_train,

        epochs=50,

        batch_size=32,

        validation_split=0.2,

        callbacks=[
            early_stop,
            reduce_lr
        ],

        verbose=1
    )

    # =========================================
    # PREDICT
    # =========================================

    pred = model.predict(
        X_test,
        verbose=0
    )

    predictions.append(pred)

# =====================================================
# VMD RECONSTRUCTION
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
# RESHAPE
# =====================================================

y_true = y_true.reshape(-1, 1)

pred_y = pred_y.reshape(-1, 1)

# =====================================================
# INVERSE SCALING
# =====================================================

y_true_real = scaler_y.inverse_transform(
    y_true
)

pred_y_real = scaler_y.inverse_transform(
    pred_y
)
# =====================================================
# NIGHT FILTER
# =====================================================

solar_elevation = test["global_radiation"].values[
    WINDOW_SIZE:
]

solar_elevation = solar_elevation[:min_len]

pred_y_real[
    solar_elevation <= 0
] = 0
# =====================================================
# REMOVE NEGATIVE
# =====================================================

pred_y_real[
    pred_y_real < 0
] = 0

# =====================================================
# METRICS
# =====================================================

mae = mean_absolute_error(
    y_true_real,
    pred_y_real
)

rmse = np.sqrt(
    mean_squared_error(
        y_true_real,
        pred_y_real
    )
)

r2 = r2_score(
    y_true_real,
    pred_y_real
)

# =====================================================
# RESULTS
# =====================================================

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R2   : {r2:.4f}")

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

print("\n===== SAMPLE PREDICTIONS =====")

for i in range(10):

    print(

        f"REAL={y_true_real[i][0]:.2f} kW | "

        f"PRED={pred_y_real[i][0]:.2f} kW"
    )