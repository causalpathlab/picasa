import pandas as pd
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt

def plot_loss(loss_f,fpath,pt_size=4.0):
	import matplotlib.pylab as plt
	plt.rcParams.update({'font.size': 20})
 
	data = pd.read_csv(loss_f)
	num_subplots = len(data.columns)
 
	if num_subplots>1:
		fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6*num_subplots), sharex=True)

		for i, column in enumerate(data.columns):
			data[[column]].plot(ax=axes[i], legend=None, linewidth=pt_size, marker='o') 
			axes[i].set_ylabel(column)
			axes[i].set_xlabel('epoch')
			axes[i].grid(False)

		plt.tight_layout()
	else:
		data[data.columns[0]].plot( legend=None, linewidth=pt_size, marker='o') 
	plt.savefig(fpath);plt.close()


