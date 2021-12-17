from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Concatenate
import numpy as np
import tensorflow as tf

class Model():

	def __init__(self, X_train, X_valid):
		self.X_train = X_train
		self.X_valid = X_valid

		self.lstm_encoder = Sequential ([
			LSTM(input_shape=[30, 4], units=128, activation='relu', return_sequences=True),
			LSTM(units=20)
		])  # 케라스 순차 모델 사용

		self.lstm_decoder = Sequential ([
			RepeatVector(20, input_shape = [20]),
			LSTM(units=20, return_sequences=True),
			LSTM(units=128, return_sequences=True),
			TimeDistributed(Dense(5, activation="sigmoid"))
		])

		self.lstm_ae = Sequential([self.lstm_encoder, self.lstm_decoder])
		self.lstm_ae.compile(optimizer='adam', loss='mse')
		self.lstm_encoder.summary()
		self.lstm_decoder.summary()


	def fit_model(self):
		history = self.lstm_ae.fit(self.X_train, self.X_train, epochs=50, validation_data=(self.X_valid, self.X_valid))



model = Model([1], [1])

wt_prime_inpt = Input(shape=[4])
inputs = [wt_prime_inpt]
encoders = []

#거래할 ETF는 한개이므로
inpt = Input(shape=[30, 4])
inputs.append(inpt)
encoder = model.lstm_encoder(inpt)
encoders.append(encoder)

enc_concat = Concatenate(axis=1)(encoders)
wt_prime_concat_with_enc = Concatenate(axis=1)([wt_prime_inpt, enc_concat])

hidden1 = Dense(64, activation="relu")(wt_prime_concat_with_enc)
hidden2 = Dense(32, activation="relu")(hidden1)
output = Dense(3)(hidden2) #output레이어

model = tf.keras.models.Model(inputs=[inputs], outputs=[output])
model.summary()
