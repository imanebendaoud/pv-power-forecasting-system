import sys,os
import numpy as np
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))
from preprocessing import load_and_preprocess
from fonction_sequence import create_sequence_multi
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# MODELE SEQ2SEQ
# ========================= 
def build_seq2seq(input_shape, horizon=24):
    encoder_inputs = Input(shape=input_shape)

    # Encoder
    encoder = LSTM(128, dropout=0.2)(encoder_inputs)

    # Répéter le contexte pour horizon sorties
    repeated = RepeatVector(horizon)(encoder)

    # Decoder
    decoder = LSTM(128, return_sequences=True, dropout=0.2)(repeated)

    # Sortie : 24 valeurs
    outputs = TimeDistributed(Dense(1))(decoder)

    model = Model(encoder_inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model


# =========================
# CHARGEMENT DES DONNEES
# =========================
train, test, scaler_y = load_and_preprocess()

# Ici on compare sur la target globale energy
features = [
    "ambient_temperature","wind_speed","module_temperature",
    "tilted_radiation","global_radiation","humidity",
    "pv_lag","radiation_diff","hour_sin","hour_cos"
]

x_train, y_train = create_sequence_multi(
    train, "energy", features, window_size=24, horizon=24
)

x_test, y_test = create_sequence_multi(
    test, "energy", features, window_size=24, horizon=24
)

# reshape y pour seq2seq
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

# =========================
# ENTRAINEMENT
# =========================
model = build_seq2seq((x_train.shape[1], x_train.shape[2]))

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    x_train,y_train,epochs=60,batch_size=32,validation_split=0.2,callbacks=[early_stop]
)
# =========================
# PREDICTION
# =========================
pred_y = model.predict(x_test)

# remettre forme (samples, 24)
pred_y = pred_y.reshape(pred_y.shape[0], pred_y.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_test.flatten(), pred_y.flatten())
rmse = np.sqrt(mean_squared_error(y_test.flatten(), pred_y.flatten()))
r2 = r2_score(y_test.flatten(), pred_y.flatten())

print("SEQ2SEQ RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)