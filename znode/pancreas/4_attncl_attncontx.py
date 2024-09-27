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

sample = 'pancreas'
wdir = 'znode/pancreas/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_indrop1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_indrop2.h5ad')
batch3 = an.read_h5ad(wdir+'data/'+sample+'_indrop3.h5ad')
batch4 = an.read_h5ad(wdir+'data/'+sample+'_indrop4.h5ad')
batch5 = an.read_h5ad(wdir+'data/'+sample+'_smartseq2.h5ad')
batch6 = an.read_h5ad(wdir+'data/'+sample+'_celseq2.h5ad')
batch7 = an.read_h5ad(wdir+'data/'+sample+'_fluidigmc1.h5ad')

picasa_object = picasa.pic.create_picasa_object(
	{
	 'indrop1':batch1,
	 'indrop2':batch2,
	 'indrop3':batch3,
	 'indrop4':batch4,
     'smartseq2':batch5,
	 'celseq2':batch6,
	 'fluidigmc1':batch7
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
		'lambda_loss' : [1.0,0.1,1.0],
'temperature_cl': 1.0, 'neighbour_method': 'approx_50', 'pair_importance_weight': 0.1, 'corruption_rate': 0.0, 'rare_ct_mode': True, 'num_clusters': 10, 'rare_group_threshold': 0.1, 'rare_group_weight': 2.0, 'epochs': 1, 'titration': 10}


dfl = pd.read_csv(wdir+'data/pancreas_label.csv.gz')
dfl = dfl[['cell','batch','celltype']]
# dfl['cell'] = dfl['index']
# dfl.rename(columns={'Cell':'cell'},inplace=True)

def plot_attention():

	import h5py as hf
	from scipy.stats import zscore
	
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

	# since we have only one pair 
	p1,p2 = batch_keys[0],batch_keys[1]

	adata_p1 = picasa_object.data.adata_list[p1]
	adata_p2 = picasa_object.data.adata_list[p2]
	nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

	
	# adata_p1.obs['celltype'] = pd.merge(pd.DataFrame(adata_p1.obs),dfl,left_index=True, right_on='cell',how='left')['celltype'].values
	 
	# adata_p2.obs['celltype'] = pd.merge(adata_p2.obs,dfl,left_index=True,right_on='cell',how='left')['celltype'].values

	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = 10
	eval_total_size = 3000
	p1_attention,p1_ylabel = picasa_object.eval_attention(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)
	
 	

	unique_celltypes = adata_p1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)
 
 
 
 
	top_genes = []
	top_n = 100
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
							index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
		df_attn = df_attn.unstack().reset_index()
		df_attn = df_attn.sort_values(0,ascending=False)
		top_genes.append(df_attn['level_0'].unique()[:top_n])
	top_genes = np.unique(np.array(top_genes).flatten())
 
 
	# cols = 3  
	# rows = int(np.ceil(num_celltypes / cols))

	# fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size as needed

	# axes = axes.flatten()
	# plt.figure(figsize=(50,50))
  
	# for idx, ct in enumerate(unique_celltypes):
	# 	ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
	# 	ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
	# 	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
	# 						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
		
	# 	df_attn = df_attn.apply(zscore)
	# 	df_attn[df_attn > 5] = 5
	# 	df_attn[df_attn < -5] = -5

	# 	df_attn = df_attn.loc[:,top_genes]
	# 	df_attn = df_attn.loc[top_genes,:]

	# 	sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
	# 	axes[idx].set_title(f"Clustermap for {ct}")

	# for j in range(idx + 1, rows * cols):
	# 	fig.delaxes(axes[j])

	# plt.tight_layout()
	# plt.savefig(wdir + 'results/sc_attention_allct.png')
	# plt.close()

	# marker = ['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB']	 

	marker = np.array(['IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A',
	'GNLY', 'NKG7', 'CST3', 'CD3E', 'FCER1A', 'CD74', 'LST1', 'CCL5',
	'HLA-DPA1', 'LDHB', 'CD79A', 'FCER1G', 'GZMB', 'S100A9',
	'HLA-DPB1', 'HLA-DRA', 'AIF1', 'CST7', 'S100A8', 'CD79B', 'COTL1',
	'CTSW', 'B2M', 'TYROBP', 'HLA-DRB1', 'PRF1', 'GZMA', 'FTL', 'NRGN'])
 
 
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
							index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5

		df_attn = df_attn.loc[:,top_genes]
		df_attn = df_attn.loc[top_genes,:]
  
		sns.clustermap(df_attn, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_attention_'+str(ct)+'.png')
		plt.close()


	# fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

	# axes = axes.flatten()

	# for idx, ct in enumerate(unique_celltypes):
	# 	ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
	# 	ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
	# 	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
	# 						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
	# 	df_attn = df_attn.apply(zscore)
	# 	df_attn[df_attn > 5] = 5
	# 	df_attn[df_attn < -5] = -5
	# 	df_attn = df_attn.loc[:,marker]
	# 	df_attn = df_attn.loc[marker,:]

	# 	sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
	# 	axes[idx].set_title(f"Clustermap for {ct}")

	# for j in range(idx + 1, rows * cols):
	# 	fig.delaxes(axes[j])

	# plt.tight_layout()
	# plt.savefig(wdir + 'results/sc_attention_allct_marker.png')
	# plt.close()

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
	eval_batch_size = 10
	eval_total_size = 3000
	p1_context,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)
	
	marker = ['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB']
	# marker = top_genes
	df_context = pd.DataFrame(np.mean(p1_context, axis=0),
							index=adata_p1.var.index.values)
	df_context = df_context.apply(zscore)
	df_context[df_context > 1] = 1
	df_context[df_context < -1] = -1
	# df_context = df_context.loc[marker,:]
	sns.clustermap(df_context, cmap='viridis')
	plt.tight_layout()
	plt.savefig(wdir + 'results/sc_context_allct.png')
	plt.close()

	unique_celltypes = adata_p1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)
	cols = 3  
	rows = int(np.ceil(num_celltypes / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

	axes = axes.flatten()

	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
							index=adata_p1.var.index.values)
		
		df_context = df_context.apply(zscore)
		df_context[df_context > 5] = 5
		df_context[df_context < -5] = -5
		df_context = df_context.loc[marker,:]

		sns.heatmap(df_context, cmap='viridis', ax=axes[idx])
		axes[idx].set_title(f"Clustermap for {ct}")

	for j in range(idx + 1, rows * cols):
		fig.delaxes(axes[j])

	plt.tight_layout()
	plt.savefig(wdir + 'results/sc_context_allct_marker.png')
	plt.close()

def plot_context_chord():
	
	import h5py as hf 
	from scipy.stats import zscore
	from sklearn.metrics.pairwise import cosine_similarity


	adata = an.read_h5ad('/data/sishir/data/pancreas/pancreas_raw.h5ad')
	genes = adata.var._index.values



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
	eval_batch_size = 10
	eval_total_size = 3000
	p1_context,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)
	
	unique_celltypes = adata_p1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)

	zcutoff = 5.0
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
		df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
							index=adata_p1.var.index.values)
  
		df_context.index = [genes[int(x)] for x in df_context.index.values]
		
		df_context.columns = ['context'+str(x) for x in range(df_context.shape[1])]
  
		df_context = df_context.apply(zscore)
		df_context = df_context[(df_context >= zcutoff) | (df_context <= -zcutoff)]
		df_context.fillna(0.0,inplace=True)
		df_context[df_context <= -zcutoff] = -1.0
		df_context[df_context >= zcutoff] = 1.0

		# high_b = np.percentile(df_context.values.flatten(),99)
		# low_b = np.percentile(df_context.values.flatten(),1)
		# df_context = df_context[(df_context >= high_b) | (df_context <= low_b)]
		# df_context.fillna(0.0,inplace=True)
		# df_context[df_context < 0.0] = 0.0
		# df_context[df_context > 0.0] = 1.0
  
  
		df_context = df_context.loc[:, df_context.sum() != 0]
		df_context.to_csv(wdir+'results/sc_context_module_'+ct+'.csv.gz',compression='gzip')
  

plot_attention()
# plot_context()

	