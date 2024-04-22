

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch
import sys
import logging

sample = 'pancreas'
wdir = 'node/pancreas/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')
distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')
distdf = distdf[['5','4112']]

device = 'cuda'
batch_size = 256
eval_batch_size=4113
input_dims = 846
emb_dim = 1000
att_dim = 5
latent_dims = 10
encoder_layers = [200,100,10]
projection_layers = [10,25,10]
corruption_rate = 0.9
l_rate = 0.001
epochs= 100

temperature = 1.0 # higher scores smooths out the differences between the similarity scores.



logging.basicConfig(filename=wdir+'results/4_attncl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

logging.info( f"Device: {device}, \
	Batch Size: {batch_size}, Evaluation Batch Size: {eval_batch_size}, \
	Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
	Encoder Layers: {encoder_layers}, Projection Layers: {projection_layers}, \
	Corruption Rate: {corruption_rate}, Learning Rate: {l_rate}, \
	Epochs: {epochs}, temperature: {temperature}")

def train():
	logging.info('train...')
	data = sailr.du.nn_load_data_triplets(rna,spatial,distdf,device,batch_size)

	features_high = int(data.dataset.sp_vals.max(axis=0).values)
	features_low = int(data.dataset.sp_vals.min(axis=0).values)

	sailr_model = sailr.nn_attn.SAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers,features_low,features_high,corruption_rate).to(device)
	logging.info(sailr_model)

	sailr.nn_attn.train(sailr_model,data,epochs,l_rate,temperature)

	torch.save(sailr_model.state_dict(),wdir+'results/nn_attncl.model')

def eval():
	
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	
	device = 'cpu'
	data_pred = sailr.du.nn_load_data_triplets(rna,spatial,distdf,device,eval_batch_size)

	features_high = int(data_pred.dataset.sp_vals.max(axis=0).values)
	features_low = int(data_pred.dataset.sp_vals.min(axis=0).values)

	sailr_model = sailr.nn_attn.SAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers,features_low,features_high,corruption_rate).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	m,ylabel = sailr.nn_attn.predict(sailr_model,data_pred)

	attn_mtx = m.attn_sc.cpu().detach().numpy()
	selected_index = [x for x,y in enumerate(rna.obs.celltype.values) if y == 'Mono/Macro']
	df_attn = pd.DataFrame(attn_mtx[selected_index].mean(0))
	df_attn.columns = rna.var.index.values
	df_attn.index = rna.var.index.values
	
	sc = [ x for x,y in enumerate(df_attn.sum(0).values>2.0) if y == True]
	df_attn = df_attn.iloc[:,sc] 
	df_attn = df_attn.iloc[sc,:] 

	sns.clustermap(df_attn)
	plt.savefig(wdir+'results/attn_marker.png')
	plt.close()

	def plot_latent(mtx,label):
		dfh = pd.DataFrame(mtx)
		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.2,metric='cosine').fit(dfh)

		df_umap= pd.DataFrame()
		df_umap['cell'] = ylabel
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		# df_umap['celltype'] = [x.split('_')[2] for x in df_umap['cell']]

		dfl = pd.read_csv(wdir+'data/pancreas_meta.tsv',sep='\t')
		dfl = dfl[['Cell','Celltype (major-lineage)']]
		dfl.columns = ['cell','celltype']
		df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values

		plot_umap_df(df_umap,'celltype',wdir+'results/nn_pcl_'+label,pt_size=1.0,ftype='png')


	plot_latent(m.h_sc.cpu().detach().numpy(),'h_sc')
	plot_latent(m.h_spp.cpu().detach().numpy(),'h_spp')

	plot_latent(m.z_sc.cpu().detach().numpy(),'z_sc')
	plot_latent(m.z_spp.cpu().detach().numpy(),'z_spp')

 

train()
eval()