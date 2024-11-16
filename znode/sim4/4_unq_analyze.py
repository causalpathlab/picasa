

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
import umap 
from picasa.util.plots import plot_umap_df

sample = 'sim4'
wdir = 'znode/sim4/'
 
df_c = pd.read_csv(wdir+'results/df_c.csv.gz',index_col=0)
df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/sim4_label.csv.gz')
dfl = dfl[['index','Cell','Batch','Group','Condition']]
dfl.columns = ['index','cell','batch','celltype','condition']



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_c)
df_umap= pd.DataFrame()
df_umap['cell'] = df_c.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_c',pt_size=1.0,ftype='png')
df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')
df_umap['condition'] = pd.merge(df_umap,dfl,on='cell',how='left')['condition'].values
plot_umap_df(df_umap,'condition',wdir+'results/nn_attncl_lat_c_batch',pt_size=1.0,ftype='png')



umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u)
df_umap= pd.DataFrame()
df_umap['cell'] = df_u.index.values
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq',pt_size=1.0,ftype='png')

df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')
df_umap['condition'] = pd.merge(df_umap,dfl,on='cell',how='left')['condition'].values
plot_umap_df(df_umap,'condition',wdir+'results/nn_attncl_lat_unq_batch',pt_size=1.0,ftype='png')


