import keras.layers
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Concatenate
import numpy as np
import tensorflow as tf
from tensorflow.python.eager.def_function import RUN_FUNCTIONS_EAGERLY, run_functions_eagerly
from keras.utils.all_utils import plot_model


class Model:

    def __init__(self):
        input_1 = Input(shape=(30, 4), name='input_1')
        input_2 = Input(shape=(2), name='input_2')

        hidden1_lstm = LSTM(30, return_sequences=True)(input_1)
        hidden2_lstm = LSTM(30, return_sequences=False)(hidden1_lstm)

        concat = Concatenate()([input_2, hidden2_lstm])

        hidden1_DNN = Dense(32)(concat)
        hidden2_DNN = Dense(32)(hidden1_DNN)
        output = Dense(3)(hidden2_DNN)

        self.model = keras.Model(inputs=[input_1, input_2], outputs=[output])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

