import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing_vmd1 import load_data_vmd
from fonction_sequence import create_sequence_multi
from tensorflow.keras.models import load_model

# =========================
# LOAD DATA
# =========================
train, test, scaler_y, scaler_std, scaler_minmax = load_data_vmd()

WINDOW = 48
HORIZON = 24

# =========================
# LOAD MODELS + FEATURES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

top_modes = np.load(
    os.path.join(BASE_DIR, "..", "top_modes.npy"),
    allow_pickle=True
).tolist()

top_modes.append("res")

feature_selection = joblib.load("artifacts/feature_selection.pkl")

# =========================
# EVALUATION STORAGE
# =========================
pred_all = []
true_all = []

# =========================
# LOOP MODES
# =========================
for m in top_modes:

    mode_col = f"mode_{m}" if m != "res" else "mode_res"

    features = feature_selection.get(mode_col, [])
    features = [f for f in features if f in test.columns]

    if len(features) == 0:
        continue

    X_test, y_test = create_sequence_multi(
        test,
        mode_col,
        features,
        WINDOW,
        HORIZON
    )

    if len(X_test) == 0:
        continue

    model = load_model(f"artifacts/models/model_{mode_col}.keras")

    y_pred = model.predict(X_test, verbose=0)

    pred_all.append(y_pred)
    true_all.append(y_test)

# =========================
# FINAL RECONSTRUCTION
# =========================
y_pred = np.sum(pred_all, axis=0)
y_true = np.mean(true_all, axis=0)

# inverse scaling
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()
y_true = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()

# clean
y_pred = np.nan_to_num(y_pred)
y_true = np.nan_to_num(y_true)

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n================ EVALUATION ================")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
print("============================================")