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


adata = an.read_h5ad('/data/sishir/data/pancreas/pancreas_raw.h5ad')
genes = adata.var._index.values

batch1.var.index = [genes[int(x)] for x in batch1.var.index.values]

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
		'lambda_loss' : [0.5,0.1,2.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
     	'corruption_rate' : 0.0,
		'epochs': 1,
		'titration': 10
		}  

def train():
	
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
	
	dfl = pd.read_csv(wdir+'data/'+sample+'_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.5,n_neighbors=20,metric='cosine').fit(df)
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
	plt.figure(figsize=(50,50))
  
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

	marker = [
    "GCG",  "MAFA",      # Alpha cells
    "INS", "PDX1", "MAFB",    # Beta cells
    "SST", "HHEX",             # Delta cells
    "PPY",      # PP cells
    "GHRL",  "SOX4",  # Epsilon cells
    "AMY2A"                    # Acinar cells
]
	# "['ARX', 'NKX6-1', 'ISL1', 'FOXP1', 'NKX2-2']
	# marker = top_genes
 
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
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
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
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
	eval_batch_size = int(batch1.shape[0]/5)
	p1_emb,p1_context,p1_context_pooled,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,device)
	
	marker =['IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A',
	   'GNLY', 'NKG7', 'CST3', 'CD3E', 'FCER1A', 'CD74', 'LST1', 'CCL5',
	   'HLA-DPA1', 'LDHB', 'CD79A', 'FCER1G', 'GZMB', 'S100A9',
	   'HLA-DPB1', 'HLA-DRA', 'AIF1', 'CST7', 'S100A8', 'CD79B', 'COTL1',
	   'CTSW', 'B2M', 'TYROBP', 'HLA-DRB1', 'PRF1', 'GZMA', 'FTL', 'NRGN']
 
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
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=50,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

 
	dfl = pd.read_csv(wdir+'data/pancreas_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
	dfl['cell'] = [x +'_'+y for x,y in zip(dfl['batch'],dfl['cell'])]
 

	pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
 
	df_umap['celltype'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
	df_umap['batch'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['batch'].values
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 

def calc_score(true_labels,cluster_labels):

	from sklearn.metrics import normalized_mutual_info_score
	from sklearn.metrics.cluster import adjusted_rand_score
	from collections import Counter

	cluster_set = set(cluster_labels)
	total_correct = sum(max(Counter(true_labels[i] for i, cl in enumerate(cluster_labels) if cl == cluster).values()) 
                        for cluster in cluster_set)
	purity = total_correct / len(true_labels)

	nmi =  normalized_mutual_info_score(true_labels,cluster_labels)
	ari = adjusted_rand_score(true_labels,cluster_labels)

	return (purity,nmi,ari)


def kmeans_cluster(df,k):
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
		return kmeans.labels_

	
def get_score():
	import h5py as hf
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	dfmain = pd.DataFrame()
	for batch in batch_keys:
		df_c = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		df_c.index = [batch+'_'+str(x) for x in df_c.index.values]
		dfmain = pd.concat([dfmain,df_c],axis=0)
  
	picasa_h5.close()


	## use std norm or quant norm 
	from sklearn.preprocessing import StandardScaler
	def standardize_row(row):
		scaler = StandardScaler()
		row_reshaped = row.values.reshape(-1, 1)  
		row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
		return pd.Series(row_standardized, index=row.index)
	dfh = dfmain.apply(standardize_row, axis=1)
	dfh.index = dfmain.index.values
	dfmain = dfh

	dfl = pd.read_csv(wdir+'data/'+sample+'_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
	dfl['cell'] = [x +'_'+y for x,y in zip(dfl['batch'],dfl['cell'])]
	celltype = pd.merge(dfmain,dfl,right_on='cell',left_index=True,how='left')['celltype'].values
	n_topics = pd.Series(celltype).nunique()
	n_topics = 8
	cluster = kmeans_cluster(dfmain.to_numpy(),n_topics)

	dfc = pd.DataFrame()
	dfc['celltype'] = celltype 
	dfc['cluster'] = cluster
	dfc['celltype'].value_counts()
	sel_ct = dfc.celltype.value_counts()[:5].index.values
	dfc = dfc.loc[dfc.celltype.isin(sel_ct)]
	print(calc_score(dfc.celltype.values,dfc.cluster.values))




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
	eval_batch_size = int(batch1.shape[0]/5)
	p1_emb,p1_context,p1_context_pooled,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,device)
	
	unique_celltypes = adata_p1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)

	zcutoff = 5.0
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[adata_p1.obs['celltype'] == ct].index.values
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
  

# train()
# eval()
# plot_latent()
# plot_scsp_overlay()
# plot_attention()
# plot_context()
get_score()

	