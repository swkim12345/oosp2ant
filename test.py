from keras.backend import truncated_normal
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

#차원 프린트
print(x.shape) #(4,3)
print(y.shape) #(4,)



x = x.reshape(x.shape[0], x.shape[1], 1)


#2 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1), return_sequences=True))
model.add(LSTM(10, activation = "relu", return_sequences=True)) #연쇄적인 LSTM 레이어를 만들기 위해 return_sequences를 True로 만들어야 함

model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=2)

x_input = array([6, 7, 8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
model.summary()
print(yhat)
