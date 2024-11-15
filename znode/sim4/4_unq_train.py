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


sample = 'sim4'
wdir = 'znode/sim4/'

directory = wdir+'/data'
pattern = 'sim4_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('sim4_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >10:
		break


file_name = file_names[0].replace('.h5ad','').replace('sim4_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,'unq',
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('sim4_','')].X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,0.0,1.0],
		'temperature_cl' : 1.0,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 0.01,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
      	'loss_clusters' : 5, 
        'loss_threshold' : 0.1, 
        'loss_weight': 2.0,
		'epochs': 1,
		'titration': 15
		}  

picasa_object.estimate_neighbour(params['pair_search_method'])	
picasa_object.set_nn_params(params)
	

enc_layers = [128,15]
unique_latent_dim = 15
dec_layers = [128,128]

picasa_object.train_unique(enc_layers,unique_latent_dim,dec_layers,l_rate=0.001,epochs=500,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='unq')

eval_batch_size = 10
eval_total_size = 10000

df_c, df_u,df_batch_id = picasa_object.eval_unique(enc_layers,unique_latent_dim,dec_layers,eval_batch_size, eval_total_size,device='cuda')
df_c.to_csv(wdir+'results/df_c.csv.gz',compression='gzip')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')
df_batch_id.to_csv(wdir+'results/df_batch_id.csv.gz',compression='gzip')




import umap 
from picasa.util.plots import plot_umap_df

sample = 'sim4'
wdir = 'znode/sim4/'
 
df_c = pd.read_csv(wdir+'results/df_c.csv.gz',index_col=0)
df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/sim4_label.csv.gz')
dfl = dfl[['index','Cell','Batch','Group']]
dfl.columns = ['index','cell','batch','celltype']

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_c',pt_size=1.0,ftype='png')
df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq',pt_size=1.0,ftype='png')

df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')


