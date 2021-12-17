import random
import numpy as np
from stock_lstm import Model

class StockAgent():
    def __init__(self):
        """
        Q network 와 target Q network 를 정의해야됨
        qnet
        qnet_target

        Replay Buffer 을 사용해야됨

        Epsilon 값 또한 정의해야됨
        """
        self.eps = 0.9

    def select_action(self, state):
        """
        epsilon greedy 를 이용하여 action을 정함
        """

        coin = random.random()

        if coin < self.eps:  # epsilon - greedy
            random_action = random.randint(-1, 1)
            return random_action
        else:
            act_values = Model.predict(state)
            return np.argmax(act_values[0])
            # 높은 q value 를 가지는 action 을 택함.





