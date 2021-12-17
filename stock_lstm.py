import keras.layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Concatenate
import numpy as np
import tensorflow as tf
from stock_environment import StockWorld

class Model():

    def __init__(self):

        self.lstm_encoder = Sequential([
            LSTM(input_shape=[30, 4], units=128, activation='relu', return_sequences=True),
            LSTM(units=30)
        ])  # 케라스 순차 모델 사용

        input_1 = keras.layers.Input(shape=[2])
        input_2 = keras.layers.Input(shape=[30, 4])
        encoder = self.lstm_encoder(input_2)

        input_1_concat_input_2 = keras.layers.Concatenate(axis=1)([input_1, encoder])

        self.lstm_decoder = Sequential([
            RepeatVector(32, input_shape=[32]),
            Dense(units = 20),
            Dense(units = 128),
            TimeDistributed(Dense(3, activation="sigmoid"))
        ])

        output = self.lstm_decoder(input_1_concat_input_2)

        self.lstm_ae = keras.Model(inputs = [input_1, input_2], outputs = [output])
        self.lstm_ae.compile(optimizer='adam', loss='mse')
        self.lstm_encoder.summary()
        self.lstm_decoder.summary()
        self.lstm_ae.summary()

    def get_dataset(self, current_time):
        dataset = StockWorld(current_time)
        current_state = dataset.current_state()
        current_state = current_state[1]
        state_np = np.delete(np.array(current_state), 0, axis=1)
        data_np = state_np.astype(np.float32)
        return data_np

    def fit_model(self, X_train, X_valid):
        history = self.lstm_ae.fit(X_train, X_train, epochs=50, validation_data=(X_valid, X_valid))

