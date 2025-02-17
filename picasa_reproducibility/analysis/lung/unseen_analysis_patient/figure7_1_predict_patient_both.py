import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


sample ='lung'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_lung.h5ad')



num_batches = len(picasa_adata.obs['batch'].unique())
input_dim = adata.shape[1]
nn_params = picasa_adata.uns['nn_params']
enc_layers = [128,25]
unique_latent_dim = nn_params['latent_dim']
common_latent_dim = nn_params['latent_dim']
dec_layers = [128,128]
nn_params['device'] = 'cpu'


picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(nn_params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+'/model_results/picasa_unique.model', map_location=torch.device(nn_params['device'])))

picasa_unique_model.eval()


df_w = pd.DataFrame(picasa_unique_model.zinb_scale.weight.data.detach().cpu().numpy())

from scipy.stats import zscore

df_w = zscore(df_w, axis=0) 
df_w = df_w.T
df_w.columns = adata.var_names


####################################
df_w_top_topics = pd.read_csv('../notebooks/data/figure6_unique_add_topic_gene.csv.gz',index_col=0)

# selected_topics = df_w_top_topics['index'].values
# df_w.reset_index(inplace=True)
# df_w['index'] = ['k'+str(x) for x in df_w['index']]
# df_w = df_w.loc[df_w['index'].isin(selected_topics)]
# df_w.drop(columns={'index'},inplace=True)
# df_w.reset_index(drop=True,inplace=True)


### patient data 

df = pd.read_csv('data/tcga_lung_expr_raw.csv.gz')
df = df.set_index('Unnamed: 0')
df.columns = [x.split('_')[1] for x in df.columns]


p = [ x for x in df_w.columns if x  in df.columns]
df = df[p]
df = df.loc[:, ~df.columns.duplicated()]
df = df.div(df.sum(axis=1), axis=0) * 10000


new_adata = ad.AnnData(X=df.values)
new_adata.var_names = df.columns
new_adata.obs_names = df.index

#### transform to picasa space
df_w = df_w[p]
df_z = df.dot(df_w.T)
# df_z.columns = df_w_top_topics['index'].values
new_adata.obsm['picasa'] = df_z.values

#############add metadata 

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.neighbors(new_adata, use_rep="picasa")
sc.tl.umap(new_adata)
sc.tl.leiden(new_adata)
sc.pl.umap(new_adata,color=['leiden'] )
plt.savefig('results/picasa_space_umap.png')


new_adata.write_h5ad('results/picasa.h5ad')

