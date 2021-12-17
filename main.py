from stock_environment import StockWorld
from stock_agent import StockAgent
import datetime
from replay_buffer import ReplayBuffer

memory = ReplayBuffer()

current_time = datetime.date(2018, 1, 1)
destination = datetime.date(2019, 1, 1)

environment = StockWorld(current_time)
agent = StockAgent()

r = 0
score = 0
state = environment.reset(current_time, destination)
t, market_feature, asset, done = state
s = [t, market_feature, asset]



while not done:
    a = agent.select_action(s)

    print(a)

    if done:
        break








