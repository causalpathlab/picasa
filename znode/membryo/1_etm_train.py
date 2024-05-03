
import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import umap

import logging

sample = 'membryo_sc'
wdir = 'znode/membryo/'
rna = an.read_h5ad(wdir+'data/'+sample+'.h5ad')

device = 'cuda'
batch_size = 128
input_dims = rna.X.shape[1]
latent_dims = 10
encoder_layers = [200,100,10]
l_rate = 0.01
epochs= 200


logging.basicConfig(filename=wdir+'results/1_etm_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')




def train():
	logging.info('train.....')
	logging.info( f"Device: {device}, \
    Batch Size: {batch_size},  \
    Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
    Encoder Layers: {encoder_layers},  \
    Learning Rate: {l_rate}, \
    Epochs: {epochs}")

	data = picasa.du.nn_load_data(rna,device,batch_size)

	picasa_model = picasa.model.nn_etm.PICASANET(input_dims, latent_dims, encoder_layers).to(device)
	logging.info(picasa_model)

	l1,l2 = picasa.model.nn_etm.train(picasa_model,data,epochs,l_rate)

	torch.save(picasa_model.state_dict(),wdir+'results/nn_etm.model')

def eval():
	logging.info('eval.....')
	eval_batch_size=int(rna.X.shape[0]/5)
	data_pred = picasa.du.nn_load_data(rna,device,eval_batch_size)
	picasa_model = picasa.model.nn_etm.PICASANET(input_dims, latent_dims, encoder_layers).to(device)
	picasa_model.load_state_dict(torch.load(wdir+'results/nn_etm.model'))
	m,ylabel = picasa.model.nn_etm.predict(picasa_model,data_pred)



	### plot theta and beta
	from picasa.util.plots import plot_umap_df,plot_gene_loading

	dfh = pd.DataFrame(m.theta.cpu().detach().numpy())
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.1,metric='cosine').fit(dfh)
	df_umap= pd.DataFrame()
	df_umap['cell'] = ylabel
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	df_umap['celltype'] = rna.obs.loc[ylabel,:]['cell_type'].values

	plot_umap_df(df_umap,'celltype',wdir+'results/nn_etm',pt_size=1.0,ftype='png')

	dfbeta = pd.DataFrame(m.beta.cpu().detach().numpy())
	dfbeta.columns = rna.var.index.values
	plot_gene_loading(dfbeta,top_n=5,max_thresh=100,fname=wdir+'results/beta')

	# mats = [
	#         m.z_sc.cpu().detach().numpy(),
	#         m.theta.cpu().detach().numpy(),
	#         m.beta.cpu().detach().numpy(),
	#         ]
	# mats_name = ['z_sc','theta','beta']


	# fig, axes = plt.subplots(1, 3, figsize=(25, 15))

	# for i in range(1):
	#     for j in range(3):
	#         idx =  j
	#         ax = axes[j]  
	#         print(idx)
	#         heatmap = ax.imshow(mats[idx], cmap='viridis', aspect='auto')
	#         ax.set_title(f'{mats_name[idx]}')
	#         fig.colorbar(heatmap, ax=ax) 

	# plt.tight_layout()  
	# plt.savefig('testnn_etm.png');plt.close()

train()
# eval()