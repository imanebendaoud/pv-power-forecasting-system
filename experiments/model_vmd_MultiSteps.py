import sys,os
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))
from Feature_selection import rf_per_vmd_modes
from fonction_sequence import create_sequence_multi
from preprocessing_vmd import load_data_vmd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
top_modes=np.load('top_modes.npy',allow_pickle=True)
top_modes = list(top_modes)
top_modes.append("res")
train,test,scaler_y=load_data_vmd()
feature_selection = rf_per_vmd_modes(train.fillna(0))
import tensorflow as tf

def weighted_mse(y_true, y_pred):
    weight = 1 + 5 * tf.abs(y_true)
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))

def model_LSTM_multi(input_shape,horizon=24):
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(horizon))
    model.compile(optimizer="adam",loss=Huber(delta=1.0))
    return model


models=[]
predictions=[]
for i in top_modes:
    if i == "res":
        mode_col = "mode_res"
    else:
        mode_col=f"mode_{i}"
    features=feature_selection[mode_col]
    features = [f for f in features if f in train.columns]
    train = train.dropna()
    x_train,y_train=create_sequence_multi(train,mode_col,features,window_size=48,horizon=24)
    x_test,y_test=create_sequence_multi(test,mode_col,features,window_size=48,horizon=24)
    early_stop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
    model=model_LSTM_multi((x_train.shape[1],x_train.shape[2]))
    history=model.fit(x_train,y_train,epochs=60,batch_size=32,validation_split=0.2,callbacks=[early_stop])
    y_pred = model.predict(x_test)
    models.append(model)
    predictions.append(y_pred)
pred_y = np.sum(predictions, axis=0) 


y_test_energy = test["energy"].values
y_true = []
window_size=48
horizon=24
for i in range(len(test) - window_size - horizon):
    y_true.append(y_test_energy[i+window_size:i+window_size+horizon])

y_true = np.array(y_true)
min_len=min(len(y_true),len(pred_y))
y_true=y_true[:min_len]
pred_y=pred_y[:min_len]

y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)
pred_y_inv = scaler_y.inverse_transform(pred_y.reshape(-1,1)).reshape(pred_y.shape)
is_day = test["is_day"].values

is_day_seq = []
for i in range(len(test) - window_size - horizon):
    is_day_seq.append(is_day[i+window_size:i+window_size+horizon])

is_day_seq = np.array(is_day_seq)

# alignement
is_day_seq = is_day_seq[:min_len]

#  correction physique
pred_y_inv[is_day_seq == 0] = 0
pred_y_inv[pred_y_inv < 0.5] = 0
mea = mean_absolute_error(y_true_inv.flatten(), pred_y_inv.flatten())
rmse = np.sqrt(mean_squared_error(y_true_inv.flatten(), pred_y_inv.flatten()))
r2 = r2_score(y_true_inv.flatten(), pred_y_inv.flatten())
print("mae:",mea)
print("rmse:",rmse)
print("r2:",r2)

#graphique
'''plt.figure(figsize=(12,5))
plt.plot(y_true_inv[:,0],label="valeur reelle",linewidth=2)
plt.plot(pred_y_inv[:,0],label="prediction",linestyle="--")
plt.title("comparaison reel vs prediction")
plt.xlabel("temps")
plt.ylabel("energie pv")
plt.legend()
plt.grid()
plt.show()

fig,axes=plt.subplots(2,2,figsize=(12,8))
horizons=[0, 6, 12, 23]
for ax, h in zip(axes.flatten(),horizons):
    ax.plot(y_true_inv[:,h], label="réel", linewidth=2)
    ax.plot(pred_y_inv[:,h], "--", label="prediction")
    ax.set_title(f"t+{h}")
    ax.grid()
axes[0,0].legend()

plt.suptitle("Comparaison multi-step")
plt.tight_layout()
plt.show()'''
# choisir une zone intéressante
time_index = test.index[window_size: window_size + len(pred_y)]
min_len = min(len(time_index), len(y_true_inv), len(pred_y_inv))

time_index = time_index[:min_len]
y_true_inv = y_true_inv[:min_len]
pred_y_inv = pred_y_inv[:min_len]
import matplotlib.pyplot as plt
idx = 0

plt.figure(figsize=(10,5))

plt.plot(
    y_true_inv[idx],
    marker='o',
    linewidth=2,
    label='Réel'
)

plt.plot(
    pred_y_inv[idx],
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

plt.figure(figsize=(12,6))

for i in [0, 50, 100]:

    plt.plot(
        y_true_inv[i],
        linewidth=2,
        alpha=0.8,
        label=f"Réel {i}"
    )

    plt.plot(
        pred_y_inv[i],
        '--',
        alpha=0.8,
        label=f"Prédit {i}"
    )

plt.title("Quelques exemples de prévisions Multi-Step")
plt.xlabel("Horizon")
plt.ylabel("Puissance PV")
plt.legend()
plt.grid(True)

plt.show()
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Modèle proposé (VMD + FS) - Multi-Step : Évolution de l'apprentissage")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()