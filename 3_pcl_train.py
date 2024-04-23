

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch
import sys
import logging

sample = 'pbmc'
wdir = 'node/pbmc/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')
distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')
distdf = distdf[['0','2967']]

device = 'cuda'
batch_size = 256
eval_batch_size= rna.shape[0]
input_dims = 1000
latent_dims = 25
encoder_layers = [200,200, 100, 100, 25]
projection_layers = [25,50, 50,10]
corruption_rate = 0.3
l_rate = 0.001
epochs= 200

temperature = 1 # higher scores smooths out the differences between the similarity scores.



logging.basicConfig(filename=wdir+'results/3_pcl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

logging.info( f"Device: {device}, \
    Batch Size: {batch_size}, Evaluation Batch Size: {eval_batch_size}, \
    Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
    Encoder Layers: {encoder_layers}, Projection Layers: {projection_layers}, \
    Corruption Rate: {corruption_rate}, Learning Rate: {l_rate}, \
    Epochs: {epochs}, Temperature: {temperature}")

def train():
	logging.info('train...')
	data = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,batch_size)

	sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers).to(device)
	logging.info(sailr_model)

	sailr.nn_pcl.train(sailr_model,data,epochs,l_rate,temperature)

	torch.save(sailr_model.state_dict(),wdir+'results/nn_pcl.model')

def eval():
    
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	data_pred = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,rna.shape[0])
	sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_pcl.model'))
	m,ylabel = sailr.nn_pcl.predict(sailr_model,data_pred)

	def plot_latent(mtx,label):
		dfh = pd.DataFrame(mtx)
		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=1.0,metric='euclidean').fit(dfh)

		df_umap= pd.DataFrame()
		df_umap['cell'] = ylabel
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
		dfl.columns = ['cell','celltype','batch']
		df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values

		plot_umap_df(df_umap,'celltype',wdir+'results/nn_pcl_'+label,pt_size=3.0,ftype='png')


	plot_latent(m.z_sc.cpu().detach().numpy(),'z_sc')
	plot_latent(m.z_spp.cpu().detach().numpy(),'z_spp')

	plot_latent(m.h_sc.cpu().detach().numpy(),'h_sc')
	plot_latent(m.h_spp.cpu().detach().numpy(),'h_spp')

def eval2(rna,spatial,distdf):
    
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	data_pred = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,rna.shape[0])
	sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_pcl.model'))
	m,ylabel = sailr.nn_pcl.predict(sailr_model,data_pred)
 
	dfsc = pd.DataFrame(m.h_sc.cpu().detach().numpy())
	dfsc.index = ylabel

	# dfsp = pd.DataFrame(m.h_spp.cpu().detach().numpy())
	# sc_index = pd.merge(dfsc,rna.obs.reset_index().reset_index(),left_index=True,right_on='index')['level_0'].values
	# sp_index = distdf.iloc[sc_index]['0'].values
	# dfsp.index = [x.replace('sp_','') for x in spatial.obs.index.values[sp_index]]

	from scipy.spatial.distance import cdist

	distmat =  cdist(rna.X.todense(), spatial.X.todense())
	sorted_indices = np.argsort(distmat, axis=1)
	distdf = pd.DataFrame(sorted_indices)
	distdf = distdf.T
	distdf = distdf[[0,3307]]
	data_pred = sailr.du.nn_load_data_pairs(spatial,rna,distdf,device,spatial.shape[0])
	m,ylabel = sailr.nn_pcl.predict(sailr_model,data_pred)
	dfsp = pd.DataFrame(m.h_sc.cpu().detach().numpy())
	dfsp.index = [x.replace('sp_','') for x in ylabel]

	dfmain = pd.concat([dfsc,dfsp])

    ###################
    ####################

	## use std norm or quant norm 
	# from sklearn.preprocessing import StandardScaler
	# def standardize_row(row):
	# 	scaler = StandardScaler()
	# 	row_reshaped = row.values.reshape(-1, 1)  
	# 	row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
	# 	return pd.Series(row_standardized, index=row.index)
	# dfh = dfmain.apply(standardize_row, axis=1)
    
    ######
    
	from asappy.util.analysis import quantile_normalization
	sc_norm,sp_norm = quantile_normalization(dfsc.to_numpy(),dfsp.to_numpy())
	dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
 
    ###################
    ####################
 
	dfh.index = dfmain.index.values
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.9,metric='euclidean').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	df_umap['batch'] = [x.split('_')[2] for x in df_umap['cell']]

	dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	dfl.columns = ['cell','celltype','batch']
	df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values
 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_pcl_3',pt_size=3.0,ftype='png')
	plot_umap_df(df_umap,'batch',wdir+'results/nn_pcl_3',pt_size=3.0,ftype='png')


# train()
# eval()
eval2(rna,spatial,distdf)