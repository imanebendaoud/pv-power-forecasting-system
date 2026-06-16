from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber

def model_LSTM_multi(input_shape, horizon=24):

    model = Sequential([

        Input(shape=input_shape),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(
            64,
            return_sequences=False
        ),

        Dropout(0.2),

        Dense(horizon)

    ])

    model.compile(
        optimizer="adam",
        loss=Huber(delta=1.0)
    )

    return model