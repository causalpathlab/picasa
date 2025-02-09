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


def generate_gene_vals(df,top_n,top_genes):

	top_genes = {}
	top_topic = []
	for x in range(df.shape[0]):
		gtab = df.T.iloc[:,x].sort_values(ascending=False)[:top_n].reset_index()
		gtab.columns = ['gene','val']
		genes = gtab['gene'].values
		if x ==0:
			top_genes['k'+str(x)] = genes
			top_topic.append('k'+str(x))
		else:
			uniq = True
			for k,v in top_genes.items():
				if len(set(v).intersection(set(genes))) != 0:
					uniq = False
					
			if uniq:
				top_genes['k'+str(x)] = genes
				top_topic.append('k'+str(x))

	all_genes = np.unique(np.concatenate(list(top_genes.values())))
	df_top = df.loc[:,all_genes]
	df_top.reset_index(inplace=True)
	df_top['index'] = ['k'+str(x) for x in df_top['index']]
 
	df_f = pd.melt(df_top,id_vars='index')
	df_f.columns=['Topic','Gene','Proportion']
	df_f = df_f[df_f['Topic'].isin(top_topic)]
	print(len(df_f['Topic'].unique()))
	return df_f

def get_topic_top_genes(df_w,top_n):

	top_genes = []
	df_top = generate_gene_vals(df_w,top_n,top_genes)

	return df_top

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

top_n = 100
df_top = get_topic_top_genes(df_w,top_n)

#####################################




## filter top topics
df_w.reset_index(inplace=True)
df_w['index'] = ['k'+str(x) for x in df_w['index']]
df_w = df_w.loc[df_w['index'].isin(df_top['Topic'].unique())]
topics = df_w['index'].values
df_w.to_csv('data/figure6_unique_add_topic_gene.csv.gz',compression='gzip')

df_w.drop(columns={'index'},inplace=True)
df_w.reset_index(drop=True,inplace=True)


## filter top genes and draw heatmap
df_w = df_w.loc[:,df_top['Gene'].unique()]
ro,co = row_col_order(df_w)
df_w = df_w.loc[ro,co]

df_w.index = topics
max_thresh = 2
df_w[df_w>max_thresh] = max_thresh
df_w[df_w<-max_thresh] = -max_thresh
sns.clustermap(df_w.T,cmap='viridis')
plt.savefig( 'results/figure6_unique_add_hmap.png')
