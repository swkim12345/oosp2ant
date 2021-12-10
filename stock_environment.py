import datetime


class StockWorld():
    """
    강화학습의 Environment 에 해당되는 부분입니다.
    """
    def __init__(self, time):
        """
        state 를 구현함
        :param time: 현재 시간
        """
        self.time = time
        self.market_feature = [[0 for _ in range(4)] for _ in range(30)]  # m, n 행렬로 표현한 시장지표
        self.asset = 0  # 자산 보유 현황 : 0 없음, 1 있음

    def step(self, action):
        """
        action 을 구현한 함수
        # action 0 : 보유 유지, action -1 : 팔기, action 1 : 사기
        """

        # action 0 일 경우 aseet 변화 없음 -> 아무것도 안 함

        if action == -1:
            if self.asset == 1:  # action -1 : 자산이 있을경우 자산이 없게 됨, 자산이 없을경우 아무것도 안 함
                self.asset = 0
        elif action == 1:
            if self.asset == 0:  # action 1 : 자산이 없을경우 자산이 있게 됨, 자산이 있을경우 아무것도 안 함
                self.asset = 1

        self.time = datetime.datetime(self.time.year, self.time.month, self.time.day + 1)

    def current_state(self):
        print(self.time)
        print(self.market_feature)
        print(self.asset)


current_time = datetime.datetime(2018, 11, 10)

test1 = StockWorld(current_time)
test1.current_state()

test1.step(1)
test1.current_state()