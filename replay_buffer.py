import collections
import random
import numpy as np


class ReplayBuffer():

    def __init__(self):
        self.buffer = collections.deque(maxlen = 500)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s_prime_list.append(s_prime)
            done_mask_list.append(done_mask)

        a_list = np.array(a_list, float)
        r_list = np.array(r_list, float)
        done_mask_list = np.array(done_mask_list, float)

        return s_list, a_list, r_list, s_prime_list, done_mask_list

    def size(self):
        return len(self.buffer)


