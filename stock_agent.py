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

        self.eps = 0.95



    def select_action(self, state):
        """
        epsilon greedy 를 이용하여 action을 정함
        """

        coin = random.random()
        #input_1 : asset input_2 : market_feature
        input_1 = np.array(state[1], dtype='float32')
        input_2 = np.array(state[2], dtype='float32')
        input_1 = input_1.reshape(1, 30, 4)
        input_2 = input_2.reshape(1, 2)
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
            act_values = self.q.model.predict((input_1, input_2))
            return np.argmax(act_values[0]) - 1
            # 높은 q value 를 가지는 action 을 택함.

    def train(self, q, q_target, memory):
        for i in range(10):
            gamma = 0.98
            s, a, r, s_prime, done_mask = memory.sample(10)

            input_s1 = np.array(s[i][1], dtype='float32')
            input_s2 = np.array(s[i][2], dtype='float32')
            input_s1 = input_s1.reshape(1, 30, 4)
            input_s2 = input_s2.reshape(1, 2)

            input_s_prime1 = np.array(s_prime[i][1], dtype='float32')
            input_s_prime2 = np.array(s_prime[i][2], dtype='float32')
            input_s_prime1 = input_s_prime1.reshape(1, 30, 4)
            input_s_prime2 = input_s_prime2.reshape(1, 2)

            max_q_prime = max(q_target.model.predict((input_s_prime1, input_s_prime2)))
            target = r[i] + gamma * max_q_prime * done_mask[i]
            target = np.array(target, dtype="float32").reshape(1, 3)

            history = q_target.model.fit([input_s1, input_s2], target, batch_size=1, epochs=5, verbose=2)
            return history













