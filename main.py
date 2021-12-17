from stock_environment import StockWorld
from stock_agent import StockAgent
from replay_buffer import ReplayBuffer
from stock_dataset import stock_dataset
from stock_lstm import Model

import datetime
import numpy as np

memory = ReplayBuffer()

current_time = datetime.date(2018, 1, 1)
destination = datetime.date(2019, 1, 1)

environment = StockWorld(current_time)
agent = StockAgent()

# encoder 학습
model = Model()
dataset = stock_dataset("data.db")
set = dataset.tech_indi()[25:]
set = np.delete(set, 0, axis=1).reshape(-1, 30, 4)
print(type(set))
print(set[0])
print(set.shape)
print(set)

model.fit_model(set[:200], set[200:])


r = 0
score = 0
state = environment.reset(current_time, destination)
t, market_feature, asset, done = state
s = [t, market_feature, asset]


while not done:
    a = agent.select_action(s)

    #print(a)

    if done:
        break








