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
	if batch_count >10:
		break


file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,'unq',
	wdir)

for batch_name in picasa_object.data.adata_list.keys():
    picasa_object.data.adata_list[batch_name].obs.index = [x+'@'+batch_name for x in picasa_object.data.adata_list[batch_name].obs.index.values]

picasa_common = an.read(wdir+'results/picasa.h5ad')

dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz')
dfl = dfl[['index','cell','patient_id','cell_type','treatment_phase']]
dfl.columns = ['index','cell','batch','celltype','treatment_phase']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

## assign batch
unique_batch = 'batch'
batch_ids = {label: idx for idx, label in enumerate(picasa_common.obs[unique_batch].unique())}
picasa_common.obs[unique_batch+'_id'] = [batch_ids[x] for x in picasa_common.obs[unique_batch]]
batch_mapping = { idx:label for idx, label in zip(picasa_common.obs.index.values,picasa_common.obs[unique_batch+'_id'])}

picasa_object.set_batch_mapping(batch_mapping)

picasa_object.set_picasa_common(picasa_common)

sample = list(picasa_object.data.adata_list.keys())[0]
input_dim = picasa_object.data.adata_list[sample].X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = picasa_common.X.shape[1]
dec_layers = [128,128]

picasa_object.train_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')


picasa_object.plot_loss(tag='unq')

eval_batch_size = 10
eval_total_size = 60000

df_u = picasa_object.eval_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size, eval_total_size,device='cuda')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')


import umap 
from picasa.util.plots import plot_umap_df

sample = 'ovary'
wdir = 'znode/ovary/'
 
picasa_common = an.read(wdir+'results/picasa.h5ad')
df_c = picasa_common.to_df()

df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz')
dfl = dfl[['index','cell','patient_id','cell_type','treatment_phase']]
dfl.columns = ['index','cell','batch','celltype','treatment_phase']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_c',pt_size=1.0,ftype='png')
df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')
df_umap['treatment_phase'] = pd.merge(df_umap,dfl,on='cell',how='left')['treatment_phase'].values
plot_umap_df(df_umap,'treatment_phase',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.0,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq',pt_size=1.0,ftype='png')

df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')

df_umap['treatment_phase'] = pd.merge(df_umap,dfl,on='cell',how='left')['treatment_phase'].values
plot_umap_df(df_umap,'treatment_phase',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')


sel_p = [
'EOC136',
'EOC153',
'EOC443',
'EOC1005',
]
sel_p_cells = dfl[dfl['batch'].isin(sel_p)]['cell'].values
df_umap = df_umap[df_umap['cell'].isin(sel_p_cells)]



df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq_sel',pt_size=1.0,ftype='png')

df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch_sel',pt_size=1.0,ftype='png')

df_umap['treatment_phase'] = pd.merge(df_umap,dfl,on='cell',how='left')['treatment_phase'].values
plot_umap_df(df_umap,'treatment_phase',wdir+'results/nn_attncl_lat_unq_batch_sel',pt_size=1.0,ftype='png')

