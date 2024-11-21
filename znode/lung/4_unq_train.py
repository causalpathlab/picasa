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


sample = 'lung'
wdir = 'znode/lung/'

directory = wdir+'/data'
pattern = 'lung_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('lung_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >10:
		break


file_name = file_names[0].replace('.h5ad','').replace('lung_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,'unq',
	wdir)

# temp fix because P9 is not evaluated
picasa_object.adata_keys = ['P10', 'P1', 'P4', 'P41', 'P25', 'P6', 'P3', 'P18', 'P7', 'P17']

for batch_name in picasa_object.data.adata_list.keys():
    picasa_object.data.adata_list[batch_name].obs.index = [x+'@'+batch_name for x in picasa_object.data.adata_list[batch_name].obs.index.values]

picasa_common = an.read(wdir+'results/picasa.h5ad')

dfl = pd.read_csv(wdir+'data/NSCLC_GSE148071_CellMetainfo_table.tsv',sep='\t')

sel_cols = ['Cell', 'Celltype (malignancy)',
       'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Patient',
       'Sample']
dfl = dfl[sel_cols]
dfl.columns = ['cell','celltype_malignancy','celltype','celltype_minor','batch','sample']
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
eval_total_size = 10000

df_u = picasa_object.eval_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size, eval_total_size,device='cuda')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')


import umap 
from picasa.util.plots import plot_umap_df
import anndata as an
import pandas as pd
sample = 'lung'
wdir = 'znode/lung/'
 
picasa_common = an.read(wdir+'results/picasa.h5ad')
df_c = picasa_common.to_df()
df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)


dfl = pd.read_csv(wdir+'data/NSCLC_GSE148071_CellMetainfo_table.tsv',sep='\t')

sel_cols = ['Cell', 'Celltype (malignancy)',
       'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Patient',
       'Sample']
dfl = dfl[sel_cols]
dfl.columns = ['cell','celltype_malignancy','celltype','celltype_minor','batch','sample']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
for col in dfl.columns:
	try:
		df_umap[col] = pd.merge(df_umap,dfl,on='cell',how='left')[col].values

		plot_umap_df(df_umap,col,wdir+'results/nn_attncl_lat_c_',pt_size=1.0,ftype='png') 
	except:
		print('failed..'+col)


umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


for col in dfl.columns:
	try:
		df_umap[col] = pd.merge(df_umap,dfl,on='cell',how='left')[col].values

		plot_umap_df(df_umap,col,wdir+'results/nn_attncl_lat_unq_',pt_size=1.0,ftype='png') 
	except:
		print('failed..'+col)