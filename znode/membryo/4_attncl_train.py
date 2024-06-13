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

sample = 'membryo'
wdir = 'znode/membryo/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')

picasa_object = picasa.pic.create_picasa_object({'sc':rna,'sp':spatial},wdir)

params = {'device' : 'cuda',
		'batch_size' : 128,
		'input_dim' : rna.X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 10,
		'latent_dim' : 10,
		'encoder_layers' : [100,10],
		'projection_layers' : [10,10],
		'learning_rate' : 0.001,
		'lambda_loss' : [0.5,0.1,1.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
     	'corruption_rate' : 0.0,
		'epochs': 1,
		'titration': 40
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
	eval_batch_size = 1000
	picasa_object.eval_model(eval_batch_size,device)
	picasa_object.save()


def plot_latent(sample_size=5000):
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df

	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
 
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		sel_indexes = random.sample(range(0,picasa_h5[batch+'_latent'].shape[0]-1), sample_size)
		df = df.iloc[sel_indexes,:]

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

		if batch == 'sc':
			df_umap['celltype'] = rna.obs.loc[df.index.values,:]['cell_type'].values
		else:
			df_umap['celltype'] = spatial.obs.loc[df.index.values,:]['cell_type'].values

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

	unique_celltypes = batch.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)
	top_genes = []
	top_n = 10
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch.obs[batch.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch.var.index.values, columns=batch.var.index.values)
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
		ct_ylabel = batch.obs[batch.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch.var.index.values, columns=batch.var.index.values)
		
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
		ct_ylabel = batch.obs[batch.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch.var.index.values, columns=batch.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5

		df_attn = df_attn.loc[:,marker]
		df_attn = df_attn.loc[marker,:]
  
		sns.clustermap(df_attn, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_attention_'+ct+'.png')
		plt.close()


	fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

	axes = axes.flatten()

	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch.obs[batch.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch.var.index.values, columns=batch.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5
		df_attn = df_attn.loc[:,marker]
		df_attn = df_attn.loc[marker,:]

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
	eval_batch_size = int(batch.shape[0]/5)
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
		ct_ylabel = batch.obs[adata_p1.obs['celltype'] == ct].index.values
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
	from sklearn.preprocessing import StandardScaler
	def standardize_row(row):
		scaler = StandardScaler()
		row_reshaped = row.values.reshape(-1, 1)  
		row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
		return pd.Series(row_standardized, index=row.index)
	dfh = dfmain.apply(standardize_row, axis=1)
	dfh.index = dfmain.index.values
	# ######
	
	# from asappy.util.analysis import quantile_normalization
	# sc_norm,sp_norm = quantile_normalization(df_sc.to_numpy(),df_sp.to_numpy())
	# dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
	# dfh.index = dfmain.index.values
	###################
	####################
 
	# dfh = dfmain

	###################
	####################
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.5,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

  
	df_umap['batch'] = [x.split('_')[0] for x in df_umap['cell']]

 
	dfscl = pd.DataFrame(rna.obs['cell_type'].reset_index())
	dfscl.columns = ['cell','celltype']
	dfspl = pd.DataFrame(spatial.obs['cell_type'].reset_index())
	dfspl.columns = ['cell','celltype']
	dfl = pd.concat([dfscl,dfspl])
 
	dfm = pd.DataFrame([x.replace('sp_','').replace('sc_','') for x in df_umap['cell']],columns=['cell'])
	dflmerge = pd.merge(dfm,dfl,on='cell',how='right')
	lmap = {x:y for x,y in zip(dflmerge['cell'],dflmerge['celltype'])}
 
	df_umap['celltype'] = [ 'sp_'+lmap[x.replace('sp_','')] if 'sp_' in x else 'sc_'+lmap[x.replace('sc_','')] for x in df_umap['cell'] ]
	df_umap['celltype2'] = [ lmap[x.replace('sp_','')] if 'sp_' in x else lmap[x.replace('sc_','')] for x in df_umap['cell'] ]
 
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_a_',pt_size=.03,ftype='png')
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_a_',pt_size=1.0,ftype='png')
	plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_scsp_a_',pt_size=1.0,ftype='png')
 
 

train()
eval()
# plot_attention()
# plot_latent()
# plot_scsp_overlay()
# plot_context()

	
# 	df_umap['celltype'] = rna.obs.loc[ylabel,:]['cell_type'].values
