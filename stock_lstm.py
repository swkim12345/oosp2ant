import keras.layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Concatenate
import numpy as np
import tensorflow as tf


class Model():

    def __init__(self, X_train, X_valid):
        self.X_train = X_train
        self.X_valid = X_valid

        self.lstm_encoder = Sequential([
            LSTM(input_shape=[30, 4], units=128, activation='relu', return_sequences=True),
            LSTM(units=20)
        ])  # 케라스 순차 모델 사용

        self.lstm_decoder = Sequential([
            RepeatVector(20, input_shape=[20]),
            LSTM(units=20, return_sequences=True),
            LSTM(units=128, return_sequences=True),
            TimeDistributed(Dense(5, activation="sigmoid"))
        ])

        self.lstm_ae = Sequential([self.lstm_encoder, self.lstm_decoder])
        self.lstm_ae.compile(optimizer='adam', loss='mse')
        self.lstm_encoder.summary()
        self.lstm_decoder.summary()

    def make_model(self):  # market feature 와, asset 을 받아 q value를 예측

        wt_prime_inpt = keras.layers.Input(shape=[2])
        inputs = [wt_prime_inpt]
        encoders = []

        # 거래할 ETF는 한개이므로
        inpt = keras.layers.Input(shape=[30, 4])
        inputs.append(inpt)
        encoder = self.lstm_encoder(inpt)
        encoders.append(encoder)
        print(type(encoder))

        wt_prime_concat_with_enc = keras.layers.Concatenate(axis=1)([wt_prime_inpt, encoder])

        hidden1 = Dense(22, activation="relu")(wt_prime_concat_with_enc)
        hidden2 = Dense(32, activation="relu")(hidden1)
        output = Dense(3)(hidden2)  # output레이어

        model = tf.keras.models.Model(inputs=[inputs], outputs=[output])
        model.summary()

        return model

    def get_dataset(self, current_time):

        dataset = StockWorld(current_time)
        current_state = dataset.current_state()
        current_state = current_state[1]
        state_np = np.delete(np.array(current_state), 0, axis=1)
        data_np = state_np.astype(np.float32)
        return data_np

    def fit_model(self):
        history = self.lstm_ae.fit(self.X_train, self.X_train, epochs=50, validation_data=(self.X_valid, self.X_valid))


model = Model([1], [1]).make_model()
print(model)
