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
wdir = 'znode/ovary_ext_p/'
cdir = ''

df_umap = pd.read_csv(wdir+'results/df_umap.csv.gz')


### get raw data - test 
directory = wdir+'/test_data'
pattern = 'ovary_*.h5ad'
file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'test_data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break

file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)

df_test = pd.DataFrame()
for ad in picasa_object.adata_keys:
    df_test = pd.concat([df_test,picasa_object.data.adata_list[ad].to_df()],axis=0)


### get raw data - train
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

df_train = pd.DataFrame()
for ad in picasa_object.adata_keys:
    df_train = pd.concat([df_train,picasa_object.data.adata_list[ad].to_df()],axis=0)



#### merge test and train 

df_test.shape
df_train.shape

df_main = pd.concat([df_train,df_test],axis=0)

df_main = df_main.loc[df_umap['cell'].values,:]

# df_umap['cluster'] = ['c_'+str(x) for x in df_umap['cluster'].values]
# df_umap.sort_values('cluster',inplace=True)

# df_main = df_main.loc[df_umap['cell'].values,:]

# df_main.index = [y+'@'+x for x,y in zip(df_main.index.values,df_umap['cluster'].values)]

# import anndata as ad
# import scanpy as sc
# adata = ad.AnnData(df_main)

# adata.obs['cluster'] = [x.split('@')[0] for x in adata.obs.index.values]

# sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon")

# df_results = pd.DataFrame(adata.uns['rank_genes_groups']['names'])

# top_genes  = []
# top_num = 25
# for col in df_results.columns:
#     for gene in df_results[col].values[:top_num]:
#         if gene not in top_genes: 
#             top_genes.append(gene)

# ###### top genes 
# df_main = df_main.loc[:,top_genes]

# df_main['cluster'] = [x.split('@')[0] for x in df_main.index.values]

# sample_num= 100
# df_sampled = df_main.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_num)))

# df_sampled = df_sampled.loc[:,df_sampled.columns[:-1]]



# df_sampled = df_sampled.T

# df_sampled.columns = [x.split('@')[0] for x in df_sampled.columns]


# sns.clustermap(np.log1p(df_sampled.T),cmap="viridis")
# plt.savefig(wdir+cdir+'hmap.png')
# plt.close()

######marker genes 

from picasa.util.plots import plot_marker_genes

marker = [
    "KRT8", "KRT18", "KRT19", "MUC1", "EPCAM", 
    "KRT7", "MUC16", "DSC2", "CDH1", "SCNN1A", "SFN",
    "THY1","CD2","C1QB","VWF",
    "STAR",
]

marker = [ x for x in marker if x in df_main.columns
          ]

umap_coords = df_umap[['umap1','umap2']].values

plot_marker_genes(wdir+cdir+'epi_',df_main,umap_coords,marker,nr=4,nc=3)