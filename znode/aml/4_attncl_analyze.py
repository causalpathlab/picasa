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

sample = 'aml'
wdir = 'znode/aml/'

logging.basicConfig(filename=wdir+'results/4_attncl_train.log',
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')

def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	dfl = pd.read_csv(wdir+'data/AML_GSE116256_CellMetainfo_table.tsv',sep='\t')
	# dfl = dfl[['cell','batch','celltype']]
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.8,n_neighbors=30,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

	
		df_umap['celltype'] = pd.merge(df_umap,dfl.loc[dfl['Patient']==batch],left_on='cell',right_on='Cell',how='left')['Celltype (major-lineage)'].values
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
	
 
	# dfh = dfmain

	###################
	####################
	
	conn,cluster = picasa.ut.clust.leiden_cluster(dfh.to_numpy(),0.3)
	pd.Series(cluster).value_counts()
	
	sel_c = []
	sel_ci = []
	for i,c in enumerate(cluster):
		if c<=12: 
			sel_c.append(c)
			sel_ci.append(i)
   
	dfh = dfh.iloc[sel_ci,:]

	# conn,cluster = picasa.ut.clust.leiden_cluster(dfh.to_numpy(),0.1)
	# pd.Series(cluster).value_counts()

	# sel_c = []
	# for i,c in enumerate(cluster):
	# 	if c<=4: sel_c.append(i)
	# dfh = dfh.iloc[sel_c,:]

	md = 0.8
	nn = 30
	dist = 'euclidean'
	umap_2d = picasa.ut.analysis.run_umap(dfh.to_numpy(),use_snn=False,min_dist=md,n_neighbors=nn,distance=dist)
	# umap_2d = picasa.ut.analysis.run_umap(dfh.to_numpy(),snn_graph=conn,min_dist=0.1,n_neighbors=30)
	
	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	# df_umap['cluster'] = pd.Categorical(sel_c)

	df_umap[['umap1','umap2']] = umap_2d
 
 
	dfl = pd.read_csv(wdir+'data/AML_GSE116256_CellMetainfo_table.tsv',sep='\t')

	cell_types = {
		'B': 'B',
		'CD4Tconv': 'T',
		'CD8T': 'T',
		'EryPro': 'EryPro',
		'GMP': 'GMP',
		'HSC': 'HSC',
		'Malignant': 'Malignant',
		'Mono/Macro': 'Mono/Macro',
		'NK': 'T',
		'Plasma': 'Plasma',
		'Progenitor': 'Progenitor',
		'Promonocyte': 'Mono/Macro',
		'Tprolif': 'T'
	}

	dfl['cell_type'] = [ cell_types[x] for x in dfl['Celltype (major-lineage)']]

	# sel_cols = ['cell_type','Celltype (major-lineage)','Patient','cluster','Celltype (malignancy)']
	sel_cols = ['cell_type','Celltype (major-lineage)','Patient','Celltype (malignancy)']
 
	for col in sel_cols:
		try:
			df_umap[col.replace(' ','')] = pd.merge(df_umap,dfl,left_on='cell',right_on='Cell',how='left')[col].values
	
			plot_umap_df(df_umap,col.replace(' ',''),wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
		except:
			print('failed..'+col)

   
	df_umap.to_csv(wdir+'results/df_umap.csv.gz',index=False, compression='gzip')


plot_scsp_overlay()