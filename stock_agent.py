import random
import numpy as np
from stock_lstm import Model
from stock_dataset import stock_dataset
import stock_lstm

class StockAgent():
    def __init__(self):
        """
        Q network 와 target Q network 를 정의
        qnet
        qnet_target

        Replay Buffer 을 사용해야됨

        Epsilon 값 또한 정의해야됨
        """
        self.q = Model()
        self.qnet = Model()

        self.eps = 0.9



    def select_action(self, state):
        """
        epsilon greedy 를 이용하여 action을 정함
        """

        coin = random.random()
        #input_1 : asset input_2 : market_feature
        input_1 = np.array(state[2], dtype='float32')

        input_2 = np.array(state[1], dtype='float32')
        print("input_1")
        print(input_1)
        # for i in range(29):
        #     input_1 = np.append(input_1, state[2]).reshape(-1, 2)
        input_1 = input_1.reshape(1, 2)
        input_2 = input_2.reshape(1, 30, 4)
        input_1 = np.asarray_chkfinite(input_1)
        input_2 = np.asarray_chkfinite(input_2)
        # print("output")
        # print(input_2, input_1)
        # print(input_1.shape)
        # print(input_2.shape)
        # print(type(input_1))
        # print(type(input_2))
        # print(type(input_1[0]))
        # print(type(input_2[0]))
        # print(type(input_2[0][0]))
        # print(type(input_2[0]))


        if coin < self.eps:  # epsilon - greedy
            random_action = random.randint(-1, 1)
            return random_action
        else:
            # https://stackoverflow.com/questions/65408896/how-to-do-prediction-with-2-inputs
            # 해결방법
            act_values = self.q.lstm_ae.predict((input_1, input_2), verbose=1)
            return np.argmax(act_values[0])
            # 높은 q value 를 가지는 action 을 택함.












