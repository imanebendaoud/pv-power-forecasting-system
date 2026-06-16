import sys,os
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing import load_and_preprocess
from fonction_sequence import create_sequence

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
# ---------- Load data ----------
train, test, scaler_y = load_and_preprocess()

# ---------- Sequences ----------
features = [
    "ambient_temperature","wind_speed","module_temperature","tilted_radiation","global_radiation","humidity","pv_lag","radiation_diff","hour_sin","hour_cos","is_day",
    "temp_radiation","energy_roll_std","radiation_accel"]
joblib.dump(features, "features.save")
x_train, y_train = create_sequence(train, "energy",feature_cols=features)
x_test, y_test = create_sequence(test, "energy",feature_cols=features)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# ---------- Model ----------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history=model.fit(x_train, y_train, epochs=40, batch_size=32,validation_split=0.2, callbacks=[early_stop])

# ---------- Prediction ----------
prediction = model.predict(x_test)

# ---------- Inverse scaling ----------
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(prediction)
y_pred_inv[y_pred_inv < 0] = 0
# ---------- Metrics ----------
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
n_points = 500

plt.figure(figsize=(12,5))

plt.plot(y_test_inv[:n_points],
         label='Valeurs réelles')

plt.plot(y_pred_inv[:n_points],
         label='Valeurs prédites')

plt.title('LSTM classique - Prévision Multi-Step')
plt.xlabel('Temps')
plt.ylabel('Puissance photovoltaïque (kW)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


models = ["LSTM", "GRU-LSTM", "VMD-LSTM", "VMD-LSTM FS"]

r2_one = [0.606, 0.788, 0.863, 0.932]
r2_multi = [0.580, 0.652, 0.836, 0.849]

x = range(len(models))

plt.figure(figsize=(10,5))

plt.bar(x, r2_one, width=0.4, label="One-Step")
plt.bar([i + 0.4 for i in x], r2_multi, width=0.4, label="Multi-Step")

plt.xticks([i + 0.2 for i in x], models)
plt.ylabel("R²")
plt.title("Comparaison globale des modèles")
plt.legend()
plt.grid(axis='y')

plt.show()