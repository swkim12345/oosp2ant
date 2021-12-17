import numpy as np
import time

#from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers

def function_param() :
	a = 5
	b = 3
	c = 10
	d = 2
	e = 30
	return  a, b, c, d, e

#데이터 샘플 생성 - clear
def gen_dataset(dataset_mode, num_of_sample=500):
	#coef, bias 구하기
	a, b, c, d, e = function_param()
	coef = np.array([b, c, d, e])
	bias = a

	#random하게 testset구성
	if dataset_mode == "random" :
		np.random.seed(42)
		X = np.random.rand(num_of_sample, 4); #x1 ~ x4
		X[0:, 1:2] = X[0:, 1:2]**2
		X[0:, 2:3] = X[0:, 2:3]**3
		X[0:, 3:4] = X[0:, 3:4]**4

		y = bias + np.matmul(X, coef.transpose()) #a + bx1^1 + cx2^2 + dx3^3 + ex4^4

	#x의 range를 일정한 간격으로 뽑음
	elif dataset_mode == "x_linear" :
		linear_x = np.arange(0, 1, 1 / num_of_sample).reshape(num_of_sample, 1)
		X = np.repeat(linear_x, repeats=4, axis=1)
		X[0:, 1:2] = X[0:, 1:2]**2
		X[0:, 2:3] = X[0:, 2:3]**3
		X[0:, 3:4] = X[0:, 3:4]**4

		y = bias + np.matmul(X, coef.transpose())

	return X, y

#Keras nn으로 model 구성 - clear
def gen_sequential_model(num_of_hidden_layer, num_of_neuron_layer, actfun_hidden='relu', actfun_output='relu', optimizer_model='sgd'):
	if num_of_hidden_layer == 1 :
		model = Sequential([
			Input(4, name='input_layer'),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(1, activation=actfun_output, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
		])

	elif num_of_hidden_layer == 2 :
		model = Sequential([
			Input(4, name='input_layer'),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(1, activation=actfun_output, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
		])

	elif num_of_hidden_layer == 3 :
		model = Sequential([
			Input(4, name='input_layer'),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(num_of_neuron_layer, activation=actfun_hidden, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
			Dense(1, activation=actfun_output, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
		])

	model.summary()

	#optimizer : sgd / adam
	model.compile(optimizer=optimizer_model, loss='mse')

	return model

#keras nn을 학습시키는 함수 - clear
def fit_sequential_model(X, y, model, epochs=200, verbose=0, validation_split=0.3):
	history = model.fit(X, y, epochs=epochs, verbose=verbose, validation_split=validation_split)
	return history


#학습 결과 출력 / 분석 함수()
def result_sequential_model(history, save_name):
	import matplotlib.pyplot as plt

	plt.figure(figsize=(15,10))

	plt.plot(history.history['loss'][1:])
	plt.plot(history.history['val_loss'][1:])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig(save_name, dpi=300)
	print("train loss=", history.history['loss'][-1])
	print("test loss=", history.history['val_loss'][-1])

#학습된 Model에 임의의 입력 변수를 입력으로 받아서 y값을 예측하는 함수
def predict_new_sample(model, x_raw):
	a, b, c, d, e = function_param()
	x = np.array([x_raw[0], x_raw[1]**2, x_raw[2]**3, x_raw[3]**4]).reshape(1,4)
	y_predict = model.predict(x)[0][0] #tf.data.iterator 반환(여기선 1개만 반환 -> [0][0]으로 접근)

	y_actual = a + b * x[0][0] + c * x[0][1] + d * x[0][2] + e * x[0][3]

	print("y actual value = ", y_actual)
	print("y predict value = ", y_predict)

#함수를 실행하는 셋
def run_nn(dataset_mode, num_of_sample, num_of_hidden_layer, num_of_node, actfun_hidden, actfun_output, \
	optimizer_model, epochs, verbose, graph_name) :

	start = time.time()
	X, y = gen_dataset(dataset_mode=dataset_mode, num_of_sample=num_of_sample)
	model = gen_sequential_model(num_of_hidden_layer=num_of_hidden_layer, num_of_neuron_layer=num_of_node,\
		 actfun_hidden=actfun_hidden, actfun_output=actfun_output, optimizer_model=optimizer_model)
	history = fit_sequential_model(X, y, model, epochs=epochs, verbose=verbose)
	end = time.time()
	print(f"{end - start} sec")
	print(type(history))
	#model 측정
	predict_new_sample(model, np.array([0.4, 0.4, 0.6, 0.6])) #샘플은 고정
	result_sequential_model(history=history, save_name=graph_name)


#조절 가능 파라미터
#gen_dataset - dataset_mode(random, x_linear), num_of_sample = 1000 / 2500 / 5000 / 10000
#gen_sequential_model - num_of_hidden_layer - 1, 2, 3 / num_of_neuron_layer = 32, 64, 128 / actfun_hidden = relu, sigmoid / actfun_output = relu, softmax / optimizer_model = adam, sgd
#fit_sequential_model = epochs = 100, 200, 400

if __name__=="__main__":

	#샘플
	dataset_mode = "x_linear"  #random, x_linear
	num_of_sample = 1000 #500, 1000, 2500
	num_of_hidden_layer = 2 # 1, 2, 3
	num_of_node = 32 # 16, 32, 64
	actfun_hidden = 'relu' #sigmoid, relu
	actfun_output = 'relu' #sigmoid, relu, softmax
	optimizer_model = 'adam' #sdg, adam
	epochs = 200 #100, 200, 400
	verbose = 2 #2로 고정

	dir = "result/"

	run_nn("random", 500, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '500_1')
	run_nn("random", 1000, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '1000')
	run_nn("random", 500, 2, 16, 'relu', 'relu', 'adam', 400, 2, dir + '500_2')

	dir = "result_2/"

	run_nn("random", 500, 3, 16, 'relu', 'relu', 'adam', 200, 2, dir + '500_1')
	run_nn("random", 1000, 3, 16, 'relu', 'relu', 'adam', 200, 2, dir + '1000')
	run_nn("random", 500, 3, 16, 'relu', 'relu', 'adam', 200, 2, dir + '500_2')

	# #reference
	# dir = 'reference/'
	# run_nn("random", 250, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	#dataset_mode 비교 - random이 훨씬 좋음
	# hidden_layer = 'relu'
	# output_layer = 'relu'
	# dir = 'dataset_xlinear/'
	# dataset = 'x_linear'
	# epoch = 200
	# run_nn(dataset, 250, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '250');
	# run_nn(dataset, 500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '500');
	# run_nn(dataset, 1000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '1000');
	# run_nn(dataset, 2500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '2500');
	# run_nn(dataset, 5000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '5000');

	# #num_of_sample 비교 layer 1
	# run_nn("random", 250, 1, 32, 'relu', 'relu', 'adam', 200, 2, 'numsample_1/250');
	# run_nn("random", 500, 1, 32, 'relu', 'relu', 'adam', 200, 2, 'numsample_1/500');
	# run_nn("random", 1000, 1, 32, 'relu', 'relu', 'adam', 200, 2, 'numsample_1/1000');
	# run_nn("random", 2500, 1, 32, 'relu', 'relu', 'adam', 200, 2, 'numsample_1/2500');
	# run_nn("random", 5000, 1, 32, 'relu', 'relu', 'adam', 200, 2, 'numsample_1/5000');

	# #reference
	# #layer 2
	# #dir = 'numsample_2/'
	# run_nn("random", 250, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	# #layer 3
	# #dir = 'numsample_3/'
	# run_nn("random", 250, 3, 32, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 3, 32, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 3, 32, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 3, 32, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 3, 32, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	# #node 16 layer 2 고정
	# # dir = 'node_16/'
	# run_nn("random", 250, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 16, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	# #node 32 layer 2 고정
	# # dir = 'node_32/'
	# run_nn("random", 250, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	# #node 64 layer 2 고정
	# # dir = 'node_64/'
	# run_nn("random", 250, 2, 64, 'relu', 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 64, 'relu', 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 64, 'relu', 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 64, 'relu', 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 64, 'relu', 'relu', 'adam', 200, 2, dir + '5000');

	# #hiddenlayer activation 변경
	# hidden_layer = 'sigmoid'
	# dir = 'hidden_sigmoid/'
	# run_nn("random", 250, 2, 32, hidden_layer, 'relu', 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, 'relu', 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, 'relu', 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, 'relu', 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, 'relu', 'adam', 200, 2, dir + '5000');

	# # output layer sigmoid
	# hidden_layer = 'relu'
	# output_layer = 'sigmoid'
	# dir = 'output_sigmoid/'
	# run_nn("random", 250, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '5000');

	# #output layer softmax
	# hidden_layer = 'relu'
	# output_layer = 'softmax'
	# dir = 'output_softmax/'
	# run_nn("random", 250, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, output_layer, 'adam', 200, 2, dir + '5000');

	# #sgd
	# hidden_layer = 'relu'
	# output_layer = 'relu'
	# dir = 'sgd/'
	# run_nn("random", 250, 2, 32, hidden_layer, output_layer, 'sgd', 200, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, output_layer, 'sgd', 200, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, output_layer, 'sgd', 200, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, output_layer, 'sgd', 200, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, output_layer, 'sgd', 200, 2, dir + '5000');

	# #epoch 100
	# hidden_layer = 'relu'
	# output_layer = 'relu'
	# dir = 'epoch_100/'
	# epoch = 100
	# run_nn("random", 250, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '5000');

	# #epoch 400
	# hidden_layer = 'relu'
	# output_layer = 'relu'
	# dir = 'epoch_400/'
	# epoch = 400
	# run_nn("random", 250, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '250');
	# run_nn("random", 500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '500');
	# run_nn("random", 1000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '1000');
	# run_nn("random", 2500, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '2500');
	# run_nn("random", 5000, 2, 32, hidden_layer, output_layer, 'adam', epoch, 2, dir + '5000');
