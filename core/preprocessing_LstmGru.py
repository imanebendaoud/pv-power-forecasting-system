import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
# Importez votre fichier sequences
from fonction_sequence import create_sequence_multi  # Remplacez par nom exact

def preprocess_simple(data_path="data_ready8.csv", window_size=48, horizon=48):
    """Simple preprocessing → sequences 48h ready pour LSTM"""
    # Load + prep basique
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index("timestamp", inplace=True)
    
    # Features cycliques
    data["hour_sin"] = np.sin(2*np.pi*data.index.hour/24)
    data["hour_cos"] = np.cos(2*np.pi*data.index.hour/24)
    
    # Resample 30min, ffill
    data = data.resample("30T").mean().fillna(method="ffill")
    
    # Features + target
    features = ["ambient_temperature", "wind_speed", "module_temperature", 
                "tilted_radiation", "global_radiation", "humidity", 
                "hour_sin", "hour_cos"]  # Ajoutez pv_lag si existant
    target = "energy"
    
    # Split temporel
    split_date = "2025-10-24 03:00:00"
    train = data[data.index <= split_date][features + [target]].copy()
    test = data[data.index > split_date][features + [target]].copy()
    
    # Scaling MinMax (uniforme)
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train), 
        index=train.index, columns=train.columns
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test), 
        index=test.index, columns=test.columns
    )
    
    # VOS fonctions sequences !
    X_train, y_train = create_sequence_multi(train_scaled, target, features, window_size, horizon)
    X_test, y_test = create_sequence_multi(test_scaled, target, features, window_size, horizon)
    
    print(f"✅ Simple ready: X_train{ X_train.shape}, y_train{ y_train.shape}")
    
    # Save tout
    pickle.dump({
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler, 'features': features,
        'window_size': window_size, 'horizon': horizon
    }, open('lstm_simple_ready.pkl', 'wb'))
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    preprocess_simple()