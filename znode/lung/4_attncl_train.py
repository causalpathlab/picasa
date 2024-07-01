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

def train():
	
	picasa_object.estimate_neighbour(params['neighbour_method'])	
	picasa_object.set_nn_params(params)
	picasa_object.train()
	picasa_object.plot_loss()

def eval():
	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = 100
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

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.1,n_neighbors=20,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		df_umap['celltype'] = pd.merge(df_umap,dfl.loc[dfl['batch']==batch],on='cell',how='left')['celltype'].values
		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')
  
		df_umap['celltype2'] = [ x.split('_')[0] for x in df_umap['celltype'].values]
		plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')

	picasa_h5.close()

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
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=1.0,n_neighbors=50,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

 
	dfl = pd.read_csv(wdir+'data/lung_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
	dfl['cell'] = [x +'_'+y for x,y in zip(dfl['batch'],dfl['cell'])]
 

	pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
 
	df_umap['celltype'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
	df_umap['batch'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['batch'].values
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')

	df_umap['celltype2'] = [ x.split('_')[0] for x in df_umap['celltype'].values]
	plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
 

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
	
	for n_topics in [4,6,8,10,12,14,16,18]:
		cluster = kmeans_cluster(dfmain.to_numpy(),n_topics)

		dfc = pd.DataFrame()
		dfc['celltype'] = celltype 
		dfc['cluster'] = cluster
  
		print(n_topics,'--',calc_score(dfc.celltype.values,dfc.cluster.values))

		# dfc['celltype'].value_counts()
		# sel_ct = dfc.celltype.value_counts()[:5].index.values
		# dfc = dfc.loc[dfc.celltype.isin(sel_ct)]
		# print(calc_score(dfc.celltype.values,dfc.cluster.values))


def plot_attention():

	import h5py as hf
	from scipy.stats import zscore
	
	### mean 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
 
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
 
	batch = batch_keys[0]
 	
	### celltype 
	ylabel = np.array([x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
 
	batch1.obs['celltype2'] = [ x.split('_')[0] for x in batch1.obs['celltype'].values]
	unique_celltypes = batch1.obs['celltype2'].unique()

	# unique_celltypes = batch1.obs['celltype'].unique()
	unique_celltypes = ['Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib']


	marker = ["CD3G", "CD8A", "SOX9", "ACTA2", "SCGB3A2", "GUCY1A1", "NKG7", "S100A8", "GNGT2", 
         "CD14", "MSLN", "MS4A2", "C1QB", "GATA3", "DCN", "ITGA8", "VCAM1", 
         "CLEC14A", "MMRN1", "PRX", "CD7", "ITGAE", "CCL17", "CD86", "CCDC113", "TOP2A", 
         "KRT5", "JCHAIN", "CD79B", "BCL11A", "SFTPB", "AGER"]
	
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype2'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5[batch+'_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		# df_attn = df_attn.apply(zscore)
		# df_attn[df_attn > 1] = 1
		# df_attn[df_attn < -1] = -1

		df_attn = df_attn.loc[:,marker]
		df_attn = df_attn.loc[marker,:]
  
		sns.clustermap(df_attn, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_attention_'+ct+'.png')
		plt.close()




# train()
# eval()
# plot_latent()
# plot_scsp_overlay()
plot_attention()
# get_score()

	