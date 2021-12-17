import matplotlib.pyplot as plt

class make_plotting():
	def __init__(self, history, asset,  save_name):
		ax1 = plt.subplot(2, 1, 1)
		plt.plot(history)
		plt.ylabel("loss")
		plt.yscale('log')
		plt.xticks(visible=False)

		ax2 = plt.subplot(2,1,2, sharex=ax1)

		plt.tight_layout()
		plt.plot(asset, '.-')
		plt.ylabel("asset")
		plt.savefig(save_name, dpi=300)

		print(len(history))
		print(len(asset))

