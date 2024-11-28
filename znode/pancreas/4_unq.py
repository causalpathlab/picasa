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

sample = 'pancreas'
wdir = 'znode/pancreas/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_indrop1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_indrop2.h5ad')
batch3 = an.read_h5ad(wdir+'data/'+sample+'_indrop3.h5ad')
batch4 = an.read_h5ad(wdir+'data/'+sample+'_indrop4.h5ad')
batch5 = an.read_h5ad(wdir+'data/'+sample+'_smartseq2.h5ad')
batch6 = an.read_h5ad(wdir+'data/'+sample+'_celseq2.h5ad')
batch7 = an.read_h5ad(wdir+'data/'+sample+'_fluidigmc1.h5ad')

picasa_object = picasa.pic.create_picasa_object(
	{
	 'indrop1':batch1,
	 'indrop2':batch2,
	 'indrop3':batch3,
	 'indrop4':batch4,
     'smartseq2':batch5,
	 'celseq2':batch6,
	 'fluidigmc1':batch7
	 },'unq',
	wdir)

params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch1.X.shape[1],
		'embedding_dim' : 2000,
		'attention_dim' : 25,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,1.0],
		'temperature_cl': 1.0, 
  		'pair_search_method': 'approx_50', 
  		'pair_importance_weight': 0.1, 
    	'corruption_tol': 5, 
     	'cl_loss_mode': 'weighted', 'loss_clusters': 10, 'loss_threshold': 0.1, 
        'loss_weight': 2.0, 'epochs': 1, 'titration': 10
}

for batch_name in picasa_object.data.adata_list.keys():
    picasa_object.data.adata_list[batch_name].obs.index = [x+'@'+batch_name for x in picasa_object.data.adata_list[batch_name].obs.index.values]

picasa_common = an.read(wdir+'results/picasa.h5ad')

## assign batch
batch_ids = {label: idx for idx, label in enumerate(picasa_common.obs['batch'].unique())}
picasa_common.obs['batch_id'] = [batch_ids[x] for x in picasa_common.obs['batch']]
batch_mapping = { idx:label for idx, label in zip(picasa_common.obs.index.values,picasa_common.obs['batch_id'])}

picasa_object.set_batch_mapping(batch_mapping)

picasa_object.set_picasa_common(picasa_common)


input_dim = picasa_object.data.adata_list['indrop1'].X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = picasa_common.X.shape[1]
dec_layers = [128,128]

picasa_object.train_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')


picasa_object.plot_loss(tag='unq')

eval_batch_size = 10
eval_total_size = 100000

df_u = picasa_object.eval_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size, eval_total_size,device='cuda')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')




import umap 
from picasa.util.plots import plot_umap_df

sample = 'pancreas'
wdir = 'znode/pancreas/'
 
picasa_common = an.read(wdir+'results/picasa.h5ad')
df_c = picasa_common.to_df()
df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/pancreas_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

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


