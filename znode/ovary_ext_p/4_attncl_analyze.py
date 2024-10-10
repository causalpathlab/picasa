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

sample = 'ovary'
wdir = 'znode/ovary_ext_p/'


def plot_scsp_overlay():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out_train_data.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	dfmain = pd.DataFrame()
	for batch in batch_keys:
		df_c = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		dfmain = pd.concat([dfmain,df_c],axis=0)
  
	picasa_h5.close()
 
	###################
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	dftest = pd.DataFrame()
	for batch in batch_keys:
		df_c = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		dftest = pd.concat([dftest,df_c],axis=0)
  
	picasa_h5.close() 
 
	########### concat train and test if needed 
	dfmain.shape
	dftest.shape

	dfmain = pd.concat([dfmain,dftest],axis=0)
	
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
	
 
	# dfh = dfmain

	###################
	####################
	
	conn,cluster = picasa.ut.clust.leiden_cluster(dfh.to_numpy(),0.1)
	pd.Series(cluster).value_counts()
	
	sel_c = []
	for i,c in enumerate(cluster):
		if c<=13: sel_c.append(i)
	dfh = dfh.iloc[sel_c,:]
  
	umap_2d = picasa.ut.analysis.run_umap(dfh.to_numpy(),use_snn=False,min_dist=0.6,n_neighbors=30)
	# umap_2d = picasa.ut.analysis.run_umap(dfh.to_numpy(),snn_graph=conn,min_dist=0.1,n_neighbors=30)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap['cluster'] = pd.Categorical(sel_c)
	df_umap[['umap1','umap2']] = umap_2d
	df_umap['batch'] = [x.split('@')[0] if '@' in x else x.split('-')[1].split('_')[0] for x in df_umap['cell'].values]

	dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz') 	 
	df_umap = pd.merge(df_umap,dfl,on='cell',how='left')
 
	plot_umap_df(df_umap,'batch_x',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 

	plot_umap_df(df_umap,'treatment_phase',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 

	df_umap['test'] = ['1TEST' if x in ['P1','P2','P3','P4'] else  '0TRAIN' for x in df_umap['batch_x']  ]

	plot_umap_df(df_umap,'test',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 

	df_umap.to_csv(wdir+'results/df_umap.csv.gz',index=False, compression='gzip')
