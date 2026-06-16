import os,sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))

from preprocessing import load_and_preprocess
from fonction_sequence import create_sequence_multi

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

#---------------load data----------------

train,test,scaler_y=load_and_preprocess()

#---------------sequences----------------
features = [
    "ambient_temperature","wind_speed","module_temperature","tilted_radiation","global_radiation","humidity","pv_lag","radiation_diff","hour_sin","hour_cos","is_day",
    "temp_radiation","energy_roll_std","radiation_accel"]

x_train,y_train=create_sequence_multi(train,"energy",features)
x_test,y_test=create_sequence_multi(test,"energy",features)

#---------------------------model-----------------------------

model=Sequential()
model.add(LSTM(64,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(24))
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history=model.fit(x_train, y_train, epochs=40, batch_size=32,validation_split=0.2, callbacks=[early_stop])
prediction = model.predict(x_test)

# -------------------- Inverse scaling ---------------------
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1,1)).reshape(y_test.shape)
y_pred_inv = scaler_y.inverse_transform(prediction.reshape(-1,1)).reshape(prediction.shape)

# ------------------------Metrics ---------------------
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# ------------------------- Plot ---------------------------

plt.plot(y_test_inv[0], label="Réel")
plt.plot(y_pred_inv[0], label="Prédit")
plt.legend()
plt.title("Prediction 24h")
plt.show()
n_points = 3

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
plt.figure(figsize=(10,5))

plt.plot(np.mean(y_test_inv[:500], axis=1), label="Réel (moyenne 24h)")
plt.plot(np.mean(y_pred_inv[:500], axis=1), label="Prédit (moyenne 24h)")

plt.title("Multi-Step - Vision globale")
plt.legend()
plt.grid()
plt.show()