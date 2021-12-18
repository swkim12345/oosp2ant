from stock_environment import StockWorld
from stock_agent import StockAgent
from replay_buffer import ReplayBuffer
from stock_dataset import stock_dataset
from stock_lstm import Model

import datetime
import numpy as np

memory = ReplayBuffer()

#시작 날짜를 받아 그 날짜부터 시작함.
current_time = datetime.date(2001, 1, 1)
destination = datetime.date(2021, 11, 30)

environment = StockWorld(current_time)
agent = StockAgent()

r = 0
score = 0
state = environment.reset(current_time, destination)
t, market_feature, asset, done = state
s = [t, market_feature, asset]

asset_list = []
history_list = []
r_list = []

while not done:
    a = agent.select_action(s)
    agent.eps = max(0.1, agent.eps - 0.0001)

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
        #loss에 대한 History를 받는 함수
        history_list.append(np.mean(agent.train(agent.q, agent.qnet, memory)))
    asset_list.append(sum(asset))
    r_list.append(r)
    print(t, r, score, asset, a)

    if (t-current_time).days % 30 == 0:
        print("parameter updated")
        agent.qnet.model.set_weights(agent.q.model.get_weights())

from make_plotting import make_plotting
print(history_list)

make_plotting(history_list, asset_list, r_list, "hello")
