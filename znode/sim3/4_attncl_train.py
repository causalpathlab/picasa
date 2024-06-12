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

sample = 'sim3'
wdir = 'znode/sim3/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_batch1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_batch2.h5ad')
batch3 = an.read_h5ad(wdir+'data/'+sample+'_batch3.h5ad')

picasa_object = picasa.pic.create_picasa_object(
    {'batch1':batch1,
     'batch2':batch2,
     'batch3':batch3
     },
    wdir)

params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch1.X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 10,
		'latent_dim' : 6,
		'encoder_layers' : [50,6],
		'projection_layers' : [6,10],
		'learning_rate' : 0.001,
		'lambda_loss' : [0.5,0.1,1.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
     	'corruption_rate' : 0.0,
		'epochs': 1,
		'titration': 24
		}  


def train():
    
	# distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')
	# scsp_map = {x:y[0] for x,y in enumerate(distdf.values)}
	# picasa_object.assign_neighbour(scsp_map,None)
	
	picasa_object.estimate_neighbour(params['neighbour_method'])
	
	picasa_object.set_nn_params(params)
	picasa_object.train()
	picasa_object.plot_loss()

def eval():
	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = int(batch1.shape[0]/5)
	picasa_object.eval_model(eval_batch_size,device)
	picasa_object.save()

def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	dfl = pd.read_csv(wdir+'data/sim3_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		df_umap['celltype'] = pd.merge(df_umap,dfl.loc[dfl['batch']==batch],on='cell',how='left')['celltype'].values
		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')

	picasa_h5.close()

def plot_attention():

	import h5py as hf
	from scipy.stats import zscore
	
	### mean 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
 
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
 
	batch = batch_keys[0]
 	
	### celltype 
	ylabel = np.array([x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

	unique_celltypes = batch1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)
	top_genes = []
	top_n = 10
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		df_attn = df_attn.unstack().reset_index()
		# df_attn = df_attn[df_attn['level_0'] != df_attn['level_1']]
		df_attn = df_attn.sort_values(0,ascending=False)
		top_genes.append(df_attn['level_0'].unique()[:top_n])
	top_genes = np.unique(np.array(top_genes).flatten())
 
 
	cols = 3  
	rows = int(np.ceil(num_celltypes / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size as needed

	axes = axes.flatten()
  
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5
		df_attn = df_attn.loc[:,top_genes]
		df_attn = df_attn.loc[top_genes,:]

		sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
		axes[idx].set_title(f"Clustermap for {ct}")

	for j in range(idx + 1, rows * cols):
		fig.delaxes(axes[j])

	plt.tight_layout()
	plt.savefig(wdir + 'results/sc_attention_allct.png')
	plt.close()

	marker =['IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A',
	   'GNLY', 'NKG7', 'CST3', 'CD3E', 'FCER1A', 'CD74', 'LST1', 'CCL5',
	   'HLA-DPA1', 'LDHB', 'CD79A', 'FCER1G', 'GZMB', 'S100A9',
	   'HLA-DPB1', 'HLA-DRA', 'AIF1', 'CST7', 'S100A8', 'CD79B', 'COTL1',
	   'CTSW', 'B2M', 'TYROBP', 'HLA-DRB1', 'PRF1', 'GZMA', 'FTL', 'NRGN']
	
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5

		df_attn = df_attn.loc[:,top_genes]
		df_attn = df_attn.loc[top_genes,:]
  
		sns.clustermap(df_attn, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_attention_'+ct+'.png')
		plt.close()


	fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

	axes = axes.flatten()

	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5
		df_attn = df_attn.loc[:,top_genes]
		df_attn = df_attn.loc[top_genes,:]

		sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
		axes[idx].set_title(f"Clustermap for {ct}")

	for j in range(idx + 1, rows * cols):
		fig.delaxes(axes[j])

	plt.tight_layout()
	plt.savefig(wdir + 'results/sc_attention_allct_marker.png')
	plt.close()

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
	
	marker =['IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A',
	   'GNLY', 'NKG7', 'CST3', 'CD3E', 'FCER1A', 'CD74', 'LST1', 'CCL5',
	   'HLA-DPA1', 'LDHB', 'CD79A', 'FCER1G', 'GZMB', 'S100A9',
	   'HLA-DPB1', 'HLA-DRA', 'AIF1', 'CST7', 'S100A8', 'CD79B', 'COTL1',
	   'CTSW', 'B2M', 'TYROBP', 'HLA-DRB1', 'PRF1', 'GZMA', 'FTL', 'NRGN']
	
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
		ct_ylabel = batch1.obs[adata_p1.obs['celltype'] == ct].index.values
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

def plot_scsp_overlay():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	dfmain = pd.DataFrame()
	for batch in batch_keys:
		df_c = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		df_c.index = [batch+'_'+str(x) for x in df_c.index.values]
		dfmain = pd.concat([dfmain,df_c],axis=0)
  
	picasa_h5.close()

	###################
	####################

	# use std norm or quant norm 
	# from sklearn.preprocessing import StandardScaler
	# def standardize_row(row):
	# 	scaler = StandardScaler()
	# 	row_reshaped = row.values.reshape(-1, 1)  
	# 	row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
	# 	return pd.Series(row_standardized, index=row.index)
	# dfh = dfmain.apply(standardize_row, axis=1)
	# dfh.index = dfmain.index.values
	# ######
	
	# from asappy.util.analysis import quantile_normalization
	# sc_norm,sp_norm = quantile_normalization(df_sc.to_numpy(),df_sp.to_numpy())
	# dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
	# dfh.index = dfmain.index.values
	###################
	####################
 
	dfh = dfmain

	###################
	####################
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

 
	dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
	dfl['cell'] = [x +'_'+y for x,y in zip(dfl['batch'],dfl['cell'])]
 

	pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
 
	df_umap['celltype'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
	df_umap['batch'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['batch'].values
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 



train()
eval()
plot_attention()
plot_latent()
# plot_scsp_overlay()
# plot_context()

	