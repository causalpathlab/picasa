import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging

sample = 'lung'
wdir = 'znode/lung/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_human.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_mouse.h5ad')

picasa_object = picasa.pic.create_picasa_object(
	{
	 'human':batch1,
	 'mouse':batch2
	 },
	wdir)


params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch1.X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 25,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,1.0,1.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
     	'corruption_rate' : 0.0,
		'epochs': 1,
		'titration': 25
		}  


def plot_context():
	
	import h5py as hf 
	from scipy.stats import zscore

	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

	# since we have only one pair 
	p1,p2 = batch_keys[0],batch_keys[1]

	adata_p1 = picasa_object.data.adata_list[p1]
	adata_p2 = picasa_object.data.adata_list[p2]
	nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = int(batch1.shape[0]/5)
	p1_emb,p1_context,p1_context_pooled,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,device)
	
	# df_context = pd.DataFrame(np.mean(p1_context, axis=0),
	# 						index=adata_p1.var.index.values)
	# df_context = df_context.apply(zscore)
	# df_context[df_context > 1] = 1
	# df_context[df_context < -1] = -1
	# # df_context = df_context.loc[marker,:]
	# sns.clustermap(df_context, cmap='viridis')
	# plt.tight_layout()
	# plt.savefig(wdir + 'results/sc_context_allct.png')
	# plt.close()

	adata_p1.obs['celltype2'] = [ x.split('_')[0] for x in adata_p1.obs['celltype'].values]
	unique_celltypes = adata_p1.obs['celltype2'].unique()
 
	unique_celltypes = ['Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib']
 
	num_celltypes = len(unique_celltypes)
	cols = 3  
	rows = int(np.ceil(num_celltypes / cols))

	# fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

	# axes = axes.flatten()

	# for idx, ct in enumerate(unique_celltypes):
	# 	ct_ylabel = batch1.obs[adata_p1.obs['celltype'] == ct].index.values
	# 	ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
	# 	df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
	# 						index=adata_p1.var.index.values)
		
	# 	df_context = df_context.apply(zscore)
	# 	df_context[df_context > 5] = 5
	# 	df_context[df_context < -5] = -5
	# 	# df_context = df_context.loc[marker,:]

	# 	sns.heatmap(df_context, cmap='viridis', ax=axes[idx])
	# 	axes[idx].set_title(f"Clustermap for {ct}")

	# for j in range(idx + 1, rows * cols):
	# 	fig.delaxes(axes[j])

	# plt.tight_layout()
	# plt.savefig(wdir + 'results/sc_context_allct_marker.png')
	# plt.close()


	zcutoff =2
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[adata_p1.obs['celltype2'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
							index=adata_p1.var.index.values)
		
		df_context = df_context.apply(zscore)
		df_context[df_context > zcutoff] = zcutoff
		df_context[df_context < -zcutoff] = -zcutoff
		# df_context = df_context.loc[marker,:]
  
		sns.clustermap(df_context, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_context_'+ct+'.png')
		plt.close()


def plot_context_chord():
	
	import h5py as hf 
	from scipy.stats import zscore
	from sklearn.metrics.pairwise import cosine_similarity


	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

	# since we have only one pair 
	p1,p2 = batch_keys[0],batch_keys[1]

	adata_p1 = picasa_object.data.adata_list[p1]
	adata_p2 = picasa_object.data.adata_list[p2]
	nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = int(batch1.shape[0]/5)
	p1_emb,p1_context,p1_context_pooled,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,device)
	
	adata_p1.obs['celltype2'] = [ x.split('_')[0] for x in adata_p1.obs['celltype'].values]
	# unique_celltypes = adata_p1.obs['celltype2'].unique()

	unique_celltypes = ['Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib']

	num_celltypes = len(unique_celltypes)

	zcutoff = 3.0
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[adata_p1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
							index=adata_p1.var.index.values)
		
		df_context.columns = ['context'+str(x) for x in range(df_context.shape[1])]
  
		df_context = df_context.apply(zscore)
		# df_context = df_context[(df_context >= zcutoff) | (df_context <= -zcutoff)]
		df_context = df_context[(df_context >= zcutoff)]
		df_context.fillna(0.0,inplace=True)

		# df_context[df_context <= -zcutoff] = -1
		# df_context[df_context >= zcutoff] = 1

		# high_b = np.percentile(df_context.values.flatten(),99)
		# low_b = np.percentile(df_context.values.flatten(),1)
		# df_context = df_context[(df_context >= high_b) | (df_context <= low_b)]
		# df_context.fillna(0.0,inplace=True)
		# df_context[df_context < 0.0] = 0.0
		# df_context[df_context > 0.0] = 1.0
    
		df_context = df_context.loc[:, df_context.sum() != 0]
		df_context.to_csv(wdir+'results/sc_context_module_'+ct+'.csv.gz',compression='gzip')
  
def klabel(k,df):
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
	return kmeans.labels_

def plot_context_chord_cluster():
	
	import h5py as hf 
	from scipy.stats import zscore
	from sklearn.metrics.pairwise import cosine_similarity


	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

	# since we have only one pair 
	p1,p2 = batch_keys[0],batch_keys[1]

	adata_p1 = picasa_object.data.adata_list[p1]
	adata_p2 = picasa_object.data.adata_list[p2]
	nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = 100
	p1_emb,p1_context,p1_context_pooled,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,device)
	adata_p1.obs['celltype2'] = [ x.split('_')[0] for x in adata_p1.obs['celltype'].values]
	# unique_celltypes = adata_p1.obs['celltype2'].unique()
	unique_celltypes = ['Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib']

	num_celltypes = len(unique_celltypes)

	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[adata_p1.obs['celltype2'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
							index=adata_p1.var.index.values)
		
		df_context.columns = ['context'+str(x) for x in range(df_context.shape[1])]
		df_context.to_csv(wdir+'results/sc_context_module_gset_'+ct+'.csv.gz',compression='gzip')

  

