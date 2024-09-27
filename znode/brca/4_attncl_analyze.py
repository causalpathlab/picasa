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


import glob
import os

sample = 'brca'
wdir = 'znode/brca/'

directory = wdir+'/data'
pattern = 'brca_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('brca_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=10:
		break


file_name = file_names[0].replace('.h5ad','').replace('brca_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('brca_','')].X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 25,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [0.5,0.1,1.0],
		'temperature_cl': 1.0, 'neighbour_method': 'approx_50', 
  		'pair_importance_weight': 0.01, 'corruption_rate': 0.0, 'rare_ct_mode': True, 'num_clusters': 10, 'rare_group_threshold': 0.1, 'rare_group_weight': 2.0, 'epochs': 1, 'titration': 50
}

def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	dfl = pd.read_csv(wdir+'data/'+sample+'_label.csv.gz')
	dfl = dfl[['cell','batch','celltype']]
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.8,n_neighbors=30,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		df_umap['celltype'] = pd.merge(df_umap,dfl.loc[dfl['batch']==batch],on='cell',how='left')['celltype'].values
		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')

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
	
	conn,cluster = picasa.ut.clust.leiden_cluster(dfh.to_numpy(),0.2)
 
	print(pd.Series(cluster).value_counts())
	
	umap_2d = picasa.ut.analysis.run_umap(dfh.to_numpy(),snn_graph=conn,min_dist=0.8,n_neighbors=30,distance='cosine')

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap['cluster'] = pd.Categorical(cluster)
	df_umap[['umap1','umap2']] = umap_2d

 
	
	dfl = pd.read_csv(wdir+'data/brca_label.csv.gz') 
 
	for col in dfl.columns[2:]:
		print(col)
		df_umap[col] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')[col].values
		df_umap[col] = pd.Categorical(df_umap[col])
		plot_umap_df(df_umap,col,wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 

plot_latent()
plot_scsp_overlay()
# get_score()

	