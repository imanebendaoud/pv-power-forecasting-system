import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from vmdpy import VMD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



#------------------------VMD------------------------------
def apply_Vmd(signal,K=3):
    u,_,_=VMD(signal,1700,0,K,0,1,1e-7)
    return u
#----------------sample entrpy-------------------
def sample_entropy(signal, m=2, r=0.2):
    signal = np.array(signal)
    N = len(signal)
    r *= np.std(signal)
    def count_similar(template_length):
        count = 0
        for i in range(N - template_length):
            template = signal[i:i+template_length]
            for j in range(i+1, N - template_length):
                window = signal[j:j+template_length]
                if np.max(np.abs(template - window)) <= r:
                    count += 1
        return count
    B = count_similar(m)
    A = count_similar(m + 1)
    if B == 0 or A == 0:
        return 0
    return -np.log(A / B)
#------------------rolling entropy-----------------
def rolling_entropy(signal, window=24, m=2, r=0.2):
    entropies = []
    signal = np.array(signal)
    for i in range(len(signal)):
        if i < window:
            entropies.append(0)
        else:
            sub_signal = signal[i-window:i]
            ent = sample_entropy(sub_signal, m, r)
            entropies.append(ent)
    return np.array(entropies)
#------------------le meilleur k pour modes------------------
'''def find_best_K(signal, K_values=[6,7,8,9,10,11,12]):
    results = {}
    for K in K_values:
        modes = apply_Vmd(signal, K)
        recon=np.sum(modes,axis=0)
        ent = sample_entropy(recon)
        results[K] = ent
        print(f"K={K} > Entropy = {ent:.4f}")
    best_K = min(results, key=results.get)
    return best_K'''


# ---------------- PREPROCESSING ----------------
def load_data_vmd(path="data_ready.csv"):

    data = pd.read_csv(path)
    
    # timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index("timestamp", inplace=True)

    # features cycliques
    data["hour_float"] = data.index.hour + data.index.minute / 60
    data["hour_sin"] = np.sin(2*np.pi*data["hour_float"]/24)
    data["hour_cos"] = np.cos(2*np.pi*data["hour_float"]/24)
    data.drop("hour_float", axis=1, inplace=True)
    # resampling FIRST (IMPORTANT FIX)
    data = data.resample("30 min").mean()
    data = data.ffill()

# THEN features
    data["pv_lag"] = data["energy"].shift(1).fillna(0)

    data["radiation_diff"] = data["global_radiation"].diff().fillna(0)

    data["is_day"] = (data["global_radiation"] > 2).astype(int)

    data["temp_radiation"] = data["module_temperature"] * data["global_radiation"]

    data["energy_roll_std"] = data["energy"].rolling(6).std().fillna(0)

    data["radiation_accel"] = data["global_radiation"].diff().diff().fillna(0)

    # split
    train = data[data.index <= "2025-10-24 03:00:00"].copy()
    test  = data[data.index >  "2025-10-24 03:00:00"].copy()
    print("train:",train.shape)
    print("test:",test.shape) 
    
    # scaling
    standard_features = ["ambient_temperature","wind_speed","module_temperature", "tilted_radiation","temp_radiation","global_radiation","pv_lag",]

    scaler_std = StandardScaler()
    train[standard_features] = scaler_std.fit_transform(train[standard_features])
    test[standard_features]  = scaler_std.transform(test[standard_features])

    scaler_MinMax = MinMaxScaler()
    train[["humidity"]] = scaler_MinMax.fit_transform(train[["humidity"]])
    test[["humidity"]]  = scaler_MinMax.transform(test[["humidity"]])

    scaler_robust = RobustScaler()
    train[["radiation_diff"]] = scaler_robust.fit_transform(train[["radiation_diff"]])
    test[["radiation_diff"]]  = scaler_robust.transform(test[["radiation_diff"]])

    scaler_y = StandardScaler()
    train["energy"] = scaler_y.fit_transform(train[["energy"]])
    test["energy"] = scaler_y.transform(test[["energy"]])
  
    #-----------------choisir le meilleur K--------------
    #best_k=find_best_K(train["energy"].values)
    #print("best k:",best_k)
    # ---------------- VMD ----------------
    train_modes = apply_Vmd(train["energy"].values, K=7)
    test_modes  = apply_Vmd(test["energy"].values, K=7)
    # =====================================
# GRAPHE DE DECOMPOSITION VMD
# =====================================

    sample_size = 1000  # nombre de points à afficher

    fig, axes = plt.subplots(8,1,figsize=(14,12),sharex=True)

# Signal original
    axes[0].plot(train["energy"].values[:sample_size])
    axes[0].set_title("Signal photovoltaïque original")

# Modes VMD
    for i in range(7):
        axes[i+1].plot(train_modes[i][:sample_size])
        axes[i+1].set_title(f"Mode {i}")

    plt.tight_layout()
    plt.show()
    variances=[np.var(train_modes[i]) for i in range(train_modes.shape[0])]# les top modes
    top_modes=np.argsort(variances)[-4:]
    print("top modes",top_modes)
    np.save("top_modes.npy",top_modes) # les sauvgarder

    min_len = min(len(train), train_modes.shape[1])
    energy=train["energy"].values[:min_len]
    modes_sum = np.sum(train_modes, axis=0)[:min_len]
    res_train=energy-modes_sum
    min_len_tes = min(len(test), test_modes.shape[1])
    energy_test=test["energy"].values[:min_len_tes]
    modes_sum_tes=np.sum(test_modes,axis=0)[:min_len_tes]
    res_test=energy_test-modes_sum_tes

    train=train.iloc[:min_len].copy()
    test=test.iloc[:min_len_tes].copy()
    train["res"]=res_train
    test["res"]=res_test

    # ---------------- ENTROPY ----------------
    #for i in range(train_modes.shape[0]):
    for i in top_modes:
        train[f"mode_{i}"] = train_modes[i][:len(train)]
        test[f"mode_{i}"] = test_modes[i][:len(test)]

        #train[f"entropy_mode_{i}"] = rolling_entropy(train_modes[i], window=24)
        #test[f"entropy_mode_{i}"]  = rolling_entropy(test_modes[i], window=24)
    train["mode_res"] = res_train
    test["mode_res"] = res_test

    #train["entropy_mode_res"] = rolling_entropy(res_train, window=24)
    #test["entropy_mode_res"] = rolling_entropy(res_test, window=24)
    sample_size = 1000

    fig, axes = plt.subplots(8,1,figsize=(15,14),sharex=True)

    axes[0].plot(
    train.index[:sample_size],
    train["energy"].values[:sample_size])
    axes[0].set_ylabel("signal original")
    #axes[0].set_title("Signal photovoltaïque original")

    for i in range(7):
        axes[i+1].plot(train.index[:sample_size],train_modes[i][:sample_size])
        axes[i+1].set_ylabel(f"M{i}")
        axes[i+1].grid(True)

    axes[-1].set_xlabel("Temps")

    plt.suptitle(
    "Décomposition VMD ",
    fontsize=14)
    plt.tight_layout()
    plt.show()
    return train,test,scaler_y
train, test, scaler_y = load_data_vmd()
