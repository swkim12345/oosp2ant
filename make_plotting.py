import matplotlib.pyplot as plt
from stock_dataset import stock_dataset

class make_plotting():
	def __init__(self, history, asset,  save_name):
		ax1 = plt.subplot(3, 1, 1)
		plt.plot(history)
		plt.ylabel("loss")
		plt.yscale('log')
		plt.xticks(visible=False)

		ax2 = plt.subplot(3,1,2)

		plt.tight_layout()
		plt.plot(asset, '.-')
		plt.ylabel("asset")
		plt.xticks(visible=False)

		#실제 코스피 값
		dataset = stock_dataset("data.db").kospi()
		ax3 = plt.subplot(3, 1,3, sharex=ax1)
		plt.plot(dataset[330:,1])

		plt.gcf().subplots_adjust(bottom=0.20)

		plt.savefig(save_name, dpi=300)
		# print(len(history))
		# print(len(asset))

		# print(dataset)
		# print(dataset[:,0])

