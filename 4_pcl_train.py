

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch
import sys
import logging

sample = 'sim_sc'
wdir = 'node/sim/'

device = 'cpu'
batch_size = 128
eval_batch_size=9984
input_dims = 20309
latent_dims = 10
encoder_layers = [200,100,10]
projection_layers = [10,25,10]
corruption_rate = 0.1
l_rate = 0.001
epochs= 100


rna = an.read_h5ad(wdir+'data/'+sample+'.h5ad')

logging.basicConfig(filename=wdir+'results/3_pcl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

logging.info( f"Device: {device}, \
    Batch Size: {batch_size}, Evaluation Batch Size: {eval_batch_size}, \
    Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
    Encoder Layers: {encoder_layers}, Projection Layers: {projection_layers}, \
    Corruption Rate: {corruption_rate}, Learning Rate: {l_rate}, \
    Epochs: {epochs}")

def train():
	logging.info('train...')
	data = sailr.du.nn_load_data(rna,device,batch_size)
	features_high = int(data.dataset.vals.max(axis=0).values)
	features_low = int(data.dataset.vals.min(axis=0).values)

	sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers,features_low,features_high,corruption_rate).to(device)
	logging.info(sailr_model)

	sailr.nn_pcl.train(sailr_model,data,epochs,l_rate)

	torch.save(sailr_model.state_dict(),wdir+'results/nn_pcl.model')

def eval():
    
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	data_pred = sailr.du.nn_load_data(rna,device,eval_batch_size)

	features_high = int(data_pred.dataset.vals.max(axis=0).values)
	features_low = int(data_pred.dataset.vals.min(axis=0).values)

	sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers,features_low,features_high,corruption_rate).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_pcl.model'))
	m,ylabel = sailr.nn_etm.predict(sailr_model,data_pred)





	def plot_latent(mtx,label):
		dfh = pd.DataFrame(mtx)
		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.5,metric='cosine').fit(dfh)

		df_umap= pd.DataFrame()
		df_umap['cell'] = ylabel
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		df_umap['celltype'] = [x.split('_')[2] for x in df_umap['cell']]

		# dfl = pd.read_csv(wdir+'data/pancreas_meta.tsv',sep='\t')
		# dfl = dfl[['Cell','Celltype (major-lineage)']]
		# dfl.columns = ['cell','celltype']
		# df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values

		plot_umap_df(df_umap,'celltype',wdir+'results/nn_pcl_'+label,pt_size=1.0,ftype='png')


	plot_latent(m.z_sc.cpu().detach().numpy(),'z_sc')
	plot_latent(m.z_scc.cpu().detach().numpy(),'z_scc')

	plot_latent(m.h_sc.cpu().detach().numpy(),'h_sc')
	plot_latent(m.h_scc.cpu().detach().numpy(),'h_scc')


train()
eval()