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

'''
초기에 설정한 목표인 "학습 시작 시점과 종료 시점간의 코스피 지수와의 차이를 비교한다. 만약 학습된 모델이 코스피 지수의 차이보다 자산의 증가를 보였다면" 이라는 목표에 대해 절반의 성공을 하였습니다. 몇개의 결론에서는 코스피에 비해 현저히 낮은 결과를 보여준 경우도 있었지만, 어떤 경우는 코스피에 비해 더 좋은 결과를 나타낸 적이 있었습니다. 다만 목표 중심이 아닌 인공신경망 모델로써 이 프로그램을 보게 된다면, loss 값이 0.1 정도로 크고, 잘 된 샘플의 경우 코스피 지수와 같은 형태를 띄어 Overfitting된 모습을 보이기 때문입니다.
'''
