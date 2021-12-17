import keras.layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Concatenate
import numpy as np
import tensorflow as tf
from tensorflow.python.eager.def_function import RUN_FUNCTIONS_EAGERLY, run_functions_eagerly

class Model():

    def __init__(self):

        input_1 = Input(shape=(30, 4), name='input_1')
        input_2 = Input(shape=(2), name='input_2')

        hidden1_lstm = LSTM(30, return_sequences=False)(input_1)

        concat = Concatenate()([input_2, hidden1_lstm])

        output = Dense(3, name='sum_output')(concat)

        self.model = keras.Model(inputs=[input_1, input_2], outputs = [output])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()
