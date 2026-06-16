import sys, os
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from Feature_selection import rf_per_vmd_modes
from preprocessing_vmd import load_data_vmd
from fonction_sequence import create_sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PARAMETRE IMPORTANT (30 min)
# =========================
WINDOW_SIZE = 48   # ✅ 24h = 48 points (30 min)

# =========================
# LOAD
# =========================
top_modes = np.load('top_modes.npy', allow_pickle=True)
top_modes = list(top_modes)
top_modes.append("res")

train, test, scaler_y = load_data_vmd()
feature_selection = rf_per_vmd_modes(train)

# =========================
# MODEL
# =========================
def model_LSTM(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# =========================
# TRAIN
# =========================
os.makedirs("saved_models", exist_ok=True)

predictions = []

for i in top_modes:

    if i == "res":
        mode_col = "mode_res"   # ✅ correction IMPORTANT
    else:
        mode_col = f"mode_{i}"

    features = feature_selection[mode_col]

    x_train, y_train = create_sequence(
        train, mode_col, feature_cols=features, window_size=WINDOW_SIZE
    )

    x_test, y_test = create_sequence(
        test, mode_col, feature_cols=features, window_size=WINDOW_SIZE
    )

    model = model_LSTM((x_train.shape[1], x_train.shape[2]))

    history=model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    model.save(f"saved_models/model_{mode_col}.h5")
    pred = model.predict(x_test)
    predictions.append(pred)
joblib.dump(feature_selection, "saved_models/feature_selection.save")
joblib.dump(scaler_y, "saved_models/scaler_y.save")
# =========================
# RECONSTRUCTION VMD
# =========================
pred_y = np.sum(predictions, axis=0)

print("taille pred:", pred_y.shape)

# =========================
# TRUE VALUES (ALIGNEMENT CORRECT)
# =========================
y_test_energy = test["energy"].values

# ✅ TRÈS IMPORTANT
y_true = y_test_energy[WINDOW_SIZE:]   # alignement avec create_sequence

# alignement final
min_len = min(len(y_true), len(pred_y))
y_true = y_true[:min_len]
pred_y = pred_y[:min_len]

# =========================
# INVERSE SCALING (IMPORTANT)
# =========================
y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1,1))
pred_inv   = scaler_y.inverse_transform(pred_y.reshape(-1,1))
# récupération du jour/nuit

is_day = test["is_day"].values[WINDOW_SIZE:]
is_day = is_day[:len(pred_inv)]

# correction physique
pred_inv[is_day == 0] = 0

pred_inv = np.maximum(pred_inv, 0)

Pmax = y_true_inv.max()

pred_inv = np.clip(pred_inv, 0, Pmax)# =========================
# METRICS (SUR DONNEES REELLES)
# =========================
mae  = mean_absolute_error(y_true_inv, pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, pred_inv))
r2   = r2_score(y_true_inv, pred_inv)

print("\nFINAL RESULTS (CORRIGÉ)")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.title("Modèle proposé (VMD + FS) - Évolution de l'apprentissage One-Step")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()