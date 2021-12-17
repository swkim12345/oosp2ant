import datetime


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
        self.destination = datetime.date(2019, 1, 1)  # 에피소드 끝
        self.time = time
        self.market_feature = [[0 for _ in range(4)] for _ in range(30)]  # m, n 행렬로 표현한 시장지표
        self.asset = [0, 1000]  # index 0 : 현금, index 1 : ETF 펀드 (kospi * 1000)

    def step(self, action):
        """
        action 을 구현한 함수
        # action 0 : 보유 유지, action -1 : 팔기, action 1 : 사기
        """

        self.time = self.time + datetime.timedelta(days=1)

        # market feature 다음 시간의 market feature 로 데이타 베이스에서 받아 최신화 시켜야 됨

        current_asset = sum(self. asset)

        if action == 0:  # 현금일 경우 아무 것도 안함, ETF 펀드일 경우 가치 최신화
            if self.asset[1] != 0:
                self.asset[1] = self.asset[1]  # 가치 최신화
        elif action == -1:
            if self.asset[1] != 0:  # ETF 펀드가 존재할 경우 현금으로 전환 수수료 계산 해주어야 됨
                self.asset[0] = self.asset[1]
                self.asset[1] = 0
        elif action == 1:
            if self.asset[0] != 0:  # 현금이 존재 할 경우 ETF 로 전환
                self.asset[1] = self.asset[0]  # 가치가 최신화 된 ETF 임 수수료 계산 해주어야 됨
                self.asset[0] = 0

        reward = (sum(self.asset) - current_asset) / current_asset

        done = self.is_done()

        return (self.time, self.market_feature, self.asset), reward, done

    def is_done(self):
        if self.time == self.destination:
            return True
        else:
            return False

    def reset(self, time):
        self.time = time
        # 나머지 state 값들도 초기화 해야 됨
        
    def current_state(self):
        """
        현재 상태를 리턴한다.
        """
        return self.time, self.market_feature, self.asset


current_time = datetime.date(2018, 11, 10)

test1 = StockWorld(current_time)
test1.current_state()

print(test1.step(-1))
print(test1.step(1))

while not test1.step(0)[2]:
    print(test1.current_state())
