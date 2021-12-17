from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

class Model():

	def __init__(self):
		self.model = Sequential(
			LSTM(64, input_shape=(30, 4), activation='relu', return_sequences=True),
			Dense(3)
		)  # 케라스 순차 모델 사용

	#0. 하이퍼 파라미터 설정하는 함수
	def set_hyperparameter():
		pass
	#1. 데이터 셋 불러오는 함수
	def load_dataset():
		pass
	#2. 모델 구성하는 함수
	def make_model():

		pass
	#3. 학습시키는 함수
	def learn_model():
		pass
	#4. 학습결과를 출력하는 함수
	def print_result():
		pass
	#5. 임의의 값을 입력받아 y값을 예측하는 함수
	def predict_value():
		pass

	#6. wrapper function
	def run_dqn(self):
		pass

if '__name__' == '__main__':
	pass
