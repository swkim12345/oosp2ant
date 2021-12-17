import datetime
from stock_dataset import stock_dataset
import numpy as np


class StockWorld():
    """
    강화학습의 Environment 에 해당되는 부분입니다.

    구현 내용

    - __init__ : 상태공간 정의
    - step(self, action) : action 에 따른 reward 와 다음 state 를 return 해주는 함수
    - is_done(self) : 에피소드가 끝났는 지 확인해 주는 함수
    - reset() : Environment 를 초기화 해주는 함수
    - current_state : 현재 상태를 return 해주는 함수
    """

    def __init__(self, time):
        """
        state 를 구현함
        :param time: 현재 시간
        """

        self.data = stock_dataset("data.db").data_modification()  # 데이터 셋 초기화
        self.kospi = stock_dataset("data.db").real_kospi()
        self.destination = datetime.date(2019, 1, 1)  # 에피소드 끝지점
        self.start = datetime.date(2000, 1, 1)  # data의 시작점

        self.time = time
        timediff = self.time - self.start

        self.market_feature = np.array([])  # m, n 행렬로 표현한 시장지표
        for feature in range(timediff.days-29, timediff.days+1):  # 시장지표 최신화
            self.market_feature = np.append(self.market_feature, np.array(self.data[feature][1:], float), axis=0)

        self.asset = np.array([1000000000, 0])  # index 0 : 현금, index 1 : ETF 펀드 (처음에 현금 10억)

    def step(self, action):
        """
        action 을 구현한 함수
        # action 0 : 보유 유지, action -1 : 팔기, action 1 : 사기
        """

        self.time = self.time + datetime.timedelta(days=1)

        timediff = self.time - self.start
        kospi_diff = float((self.kospi[timediff.days] - self.kospi[timediff.days - 1]) / self.kospi[timediff.days - 1])
        self.market_feature = np.array([])
        for feature in range(timediff.days-29, timediff.days+1):  # 시장지표 최신화
            self.market_feature = np.append(self.market_feature, np.array(self.data[feature][1:], float), axis = 0)

        current_asset = sum(self. asset)

        if action == 0:  # 현금일 경우 아무 것도 안함, ETF 펀드일 경우 가치 최신화
            if self.asset[1] != 0:
                self.asset[1] = self.asset[1] + self.asset[1] * kospi_diff # 가치 최신화
        elif action == -1:
            if self.asset[1] != 0:  # 현금일 경우 아무것도 안 함, ETF 펀드가 존재할 경우 현금으로 전환
                self.asset[0] = self.asset[1]
                self.asset[1] = 0
        elif action == 1:
            if self.asset[0] != 0:  # 현금이 존재 할 경우 ETF 로 전환, ETF 펀드가 존재 할 경우 가치 최신화
                self.asset[1] = self.asset[0] + self.asset[0] * kospi_diff  # 가치가 최신화 된 ETF 임
                self.asset[0] = 0
            else:
                self.asset[1] = self.asset[1] + self.asset[1] * kospi_diff # 가치 최신화

        reward = ((sum(self.asset) - current_asset) / current_asset) * 100

        done = self.is_done()

        np_market_feature = np.array(self.market_feature, float)
        np_asset = np.array(self.asset, float)

        return self.time, np_market_feature, np_asset, done, reward

    def is_done(self):
        if self.time == self.destination:
            return True
        else:
            return False

    def reset(self, time, destination):
        self.time = time
        self.destination = destination
        np_market_feature = np.array(self.market_feature, float)
        np_asset = np.array(self.asset, float)
        done = self.is_done()
        return self.time, np_market_feature, np_asset , done
        # 나머지 state 값들도 초기화 해야 됨
        
    def current_state(self):
        """
        현재 상태를 리턴한다.
        """
        np_market_feature = np.array(self.market_feature, float)
        np_asset = np.array(self.asset, float)
        done = self.is_done()
        return self.time, np_market_feature, np_asset, done





