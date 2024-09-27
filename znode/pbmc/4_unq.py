
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

sample = 'pbmc'
wdir = 'znode/pbmc/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_pbmc1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_pbmc2.h5ad')

picasa_object = picasa.pic.create_picasa_object(
	{'pbmc1':batch1,
	 'pbmc2':batch2
	 },
	wdir)


params = {'device': 'cuda', 'batch_size': 128, 'input_dim': 1000, 'embedding_dim': 1000, 'attention_dim': 25, 'latent_dim': 15, 'encoder_layers': [100,15], 'projection_layers': [15, 15], 'learning_rate': 0.001, 'lambda_loss': [0.5, 0.1, 1.0], 'temperature_cl': 1.0, 'neighbour_method': 'approx_50', 'pair_importance_weight': 0.0, 'corruption_rate': 0.0, 'rare_ct_mode': True, 'num_clusters': 5, 'rare_group_threshold': 0.1, 'rare_group_weight': 2.0, 'epochs': 1, 'titration': 50}

picasa_object.estimate_neighbour(params['neighbour_method'])

picasa_object.set_nn_params(params)

import h5py as hf

picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]


p1,p2 = batch_keys[0],batch_keys[1]

adata_p1 = picasa_object.data.adata_list[p1]
adata_p2 = picasa_object.data.adata_list[p2]
nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

	

train_batch_size = 64
unq_layers = [15,15,15]

picasa_object.train_unique(adata_p1,adata_p2,nbr_map,unq_layers,train_batch_size,l_rate=0.001,epochs=20)

eval_batch_size = 10
eval_total_size = 10000

df_c, df_u = picasa_object.eval_unique(adata_p1,adata_p2,nbr_map,unq_layers,eval_batch_size, eval_total_size,device='cuda')

dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
 
import umap 
from picasa.util.plots import plot_umap_df


umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_c',pt_size=1.0,ftype='png')


umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq',pt_size=1.0,ftype='png')
