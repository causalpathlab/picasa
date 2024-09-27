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
wdir = 'znode/ovary/'

directory = wdir+'/data'
pattern = 'ovary_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break


file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('ovary_','')].X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,1.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
	 	'corruption_rate' : 0.0,
		'pair_importance_weight' : 0.01,
        'rare_ct_mode' : False, 
      	'num_clusters' : 5, 
        'rare_group_threshold' : 0.1, 
        'rare_group_weight': 2.0,
		'epochs': 1,
		'titration': 10
		}  

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
unq_layers = [15,100,15]

picasa_object.train_unique(adata_p1,adata_p2,nbr_map,unq_layers,train_batch_size,l_rate=0.001,epochs=20)

eval_batch_size = 10
eval_total_size = 10000

df_c, df_u = picasa_object.eval_unique(adata_p1,adata_p2,nbr_map,unq_layers,eval_batch_size, eval_total_size,device='cuda')

dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz')
dfl = dfl[['index','cell','patient_id','cell_type']]
 
import umap 
from picasa.util.plots import plot_umap_df


umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['cell_type'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_unq_c',pt_size=1.0,ftype='png')


umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['cell_type'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_unq_u',pt_size=1.0,ftype='png')
