import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


sample ='brca'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_brca.h5ad')



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


def generate_gene_ranking(df,n_gene):

	gene_ranking = {}
	for x in df.index.values:
		h_gtab = df.T.loc[:,x].sort_values(ascending=False)[:n_gene].reset_index()
		h_gtab.columns = ['gene','val']
		h_genes = h_gtab['gene'].values
  
		gene_ranking[x] = h_genes

	return gene_ranking

def row_col_order(dfm):

	from scipy.cluster import hierarchy

	df = dfm.copy()
 
	Z = hierarchy.ward(df.to_numpy())
	ro = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, df.to_numpy()))

	df['topic'] = df.index.values
	dfm = pd.melt(df,id_vars='topic')
	dfm.columns = ['row','col','values']
	M = dfm[['row', 'col', 'values']].copy()
	M['row'] = pd.Categorical(M['row'], categories=ro)
	M = M.pivot(index='row', columns='col', values='values').fillna(0)
	co = np.argsort(-M.values.max(axis=0))
	co = M.columns[co]
 
	return ro,co


####################################

selected_topics = ['u5', 'u15','u22','u35', 'u48','u55','u96']


df_w.reset_index(inplace=True)
df_w['index'] = ['u'+str(x) for x in df_w['index']]
df_w = df_w.loc[df_w['index'].isin(selected_topics)]
df_w.drop(columns={'index'},inplace=True)
df_w.reset_index(drop=True,inplace=True)

n_gene = 10
gene_ranking = generate_gene_ranking(df_w,n_gene)
all_genes = np.unique(np.concatenate([v for k,v in gene_ranking.items()]))
#####################################


## filter top genes and draw heatmap
df_w = df_w.loc[:,all_genes]
ro,co = row_col_order(df_w)
df_w = df_w.loc[ro,co]

df_w.index = np.array(selected_topics)[ro]
max_thresh = 4
df_w[df_w>max_thresh] = max_thresh
df_w[df_w<-max_thresh] = -max_thresh
sns.clustermap(df_w.T,
               yticklabels=df_w.columns,
               xticklabels=df_w.index,
               cmap='viridis',figsize=(15,25))
plt.savefig( 'results/figure7_unique_add_hmap.pdf')
