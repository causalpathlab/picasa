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

sample = 'haber'
wdir = 'znode/haber/'


def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	dfl = pd.read_csv(wdir+'data/'+sample+'_label.csv.gz')
	# dfl = dfl[['cell','batch','celltype']]
 
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

	picasa_common = an.read_h5ad(wdir+'results/picasa.h5ad','r')
	dfmain = picasa_common.to_df()
	

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
	
	umap_2d = picasa.ut.analysis.run_umap (dfh.to_numpy(),use_snn=False,min_dist=0.5,n_neighbors=30)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values

	df_umap[['umap1','umap2']] = umap_2d
	df_umap['batch'] = [x.split('@')[1] for x in df_umap['cell'].values]

	dfl = pd.read_csv(wdir+'data/haber_label.csv.gz') 	 
	dfl.index = [x+'@'+y for x,y in zip(dfl['barcode'],dfl['batch'])]
	df_umap = pd.merge(df_umap,dfl,left_on='cell',right_index=True,how='left')


	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'batch_x',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 



# plot_latent()
plot_scsp_overlay()