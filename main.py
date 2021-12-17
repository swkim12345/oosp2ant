from stock_environment import StockWorld
from stock_agent import StockAgent
from replay_buffer import ReplayBuffer
from stock_dataset import stock_dataset
from stock_lstm import Model

import datetime
import numpy as np

memory = ReplayBuffer()

current_time = datetime.date(2001, 1, 1)
destination = datetime.date(2021, 11, 30)

environment = StockWorld(current_time)
agent = StockAgent()

r = 0
score = 0
state = environment.reset(current_time, destination)
t, market_feature, asset, done = state
s = [t, market_feature, asset]


while not done:
    a = agent.select_action(s)
    agent.eps -= 0.01

    t, market_feature, asset, done, r = environment.step(a)
    s_prime = [t, market_feature, asset]

    done_mask = 0.0 if done else 1.0

    transition = [s, a, r, s_prime, done_mask]

    memory.put(transition)
    s = s_prime
    score += r

    if done:
        break

    if memory.size() > 50:
        agent.train(agent.q, agent.qnet, memory)

    print(t, score, asset, a)








