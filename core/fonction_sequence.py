import numpy as np

# ---------- One-step ----------
import numpy as np

def create_sequence(data, target, feature_cols=None, window_size=12):
    if feature_cols is None:
        feature_cols = data.drop(columns=[target]).columns

    X_data = data[feature_cols].to_numpy(dtype=np.float32)
    y_data = data[target].to_numpy(dtype=np.float32)

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(X_data[i:i+window_size])
        y.append(y_data[i+window_size])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
# ---------- Multi-step ----------
def create_sequence_multi(data, target_col, feature_cols, window_size=48, horizon=24):

    data = data.reset_index(drop=True)
    feature_cols = [f for f in feature_cols if f in data.columns]
    X, y = [], []
    block = data[feature_cols].values
    target = data[target_col].values

    for i in range(len(data) - window_size - horizon):
        X.append(block[i:i+window_size])
        y.append(target[i+window_size:i+window_size+horizon])

    return np.array(X), np.array(y)