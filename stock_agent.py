import random
import numpy as np
from stock_lstm import Model
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
        input_1 = state[2]
        input_2 = state[1]
        for i in range(29):
            input_2 = np.append(input_1, state[2]).reshape(-1, 2)

        print(input_2, input_1)

        if coin < self.eps:  # epsilon - greedy
            random_action = random.randint(-1, 1)
            return random_action
        else:
            act_values = self.q.lstm_ae.predict([input_1, input_2], verbose=1)
            return np.argmax(act_values[0])
            # 높은 q value 를 가지는 action 을 택함.












