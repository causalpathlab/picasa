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
cdir = 'figures/fig_3_b/'

df_umap = pd.read_csv(wdir+'results/df_umap_EOC1005.csv.gz')

df_umap = df_umap.loc[df_umap['cluster'].isin(['c_4','c_5']),:]
### get raw data 
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
	batch_map,'unq',
	wdir)

df_main = pd.DataFrame()
for ad in picasa_object.adata_keys:
    df_main = pd.concat([df_main,picasa_object.data.adata_list[ad].to_df()],axis=0)


sel_patient_cells = [x.split('@')[0] for x in df_umap['cell'].values]
df_main = df_main.loc[sel_patient_cells,:]


df_main.index = [y+'@'+x for x,y in zip(df_main.index.values,df_umap['cluster'].values)]


import anndata as ad
import scanpy as sc
adata = ad.AnnData(df_main)

adata.obs['cluster'] = [x.split('@')[0] for x in adata.obs.index.values]

sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon")

df_results = pd.DataFrame(adata.uns['rank_genes_groups']['names'])

top_genes  = []
top_num = 5
for col in df_results.columns:
    for gene in df_results[col].values[:top_num]:
        if gene not in top_genes: 
            top_genes.append(gene)

###### top genes 
df_main_tg = df_main.loc[:,top_genes].copy()

df_main_tg['cluster'] = [x.split('@')[0] for x in df_main_tg.index.values]

sample_num= 100
df_sampled = df_main_tg.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_num)))

df_sampled = df_sampled.loc[:,df_sampled.columns[:-1]]



df_sampled = df_sampled.T

df_sampled.columns = [x.split('@')[0] for x in df_sampled.columns]


sns.clustermap(np.log1p(df_sampled.T),cmap="viridis")
plt.savefig(wdir+cdir+'hmap.png')
plt.close()

######marker genes 
marker = [
     "FOS", "IL6", "SOCS3", "ATF3", 
    "DUSP1", "EGR1", "FOSB", "EGR2", "TNF", 
    "HES1",  "NR4A1", "GADD45G",  
    "SNAI2", "CREB5",  "HIST1H2BC", "HIST1H2BG"
]

marker=top_genes
marker = [ x for x in marker if x in df_main.columns
          ]
df_main_mg = df_main.loc[:,marker].copy()

df_main_mg['cluster'] = [x.split('@')[0] for x in df_main_mg.index.values]

sample_num= 100
df_sampled = df_main_mg.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_num)))

df_sampled = df_sampled.loc[:,df_sampled.columns[:-1]]



df_sampled = df_sampled.T

df_sampled.columns = [x.split('@')[0] for x in df_sampled.columns]


sns.clustermap(np.log1p(df_sampled.T),cmap="viridis")
plt.savefig(wdir+cdir+'hmap_marker_tg.png')
plt.close()



umap_coords = df_umap[['umap1','umap2']].values
marker = [
     "FOS", "IL6", "SOCS3", "ATF3", 
    "DUSP1", "EGR1", "FOSB", "EGR2", "TNF", 
    "HES1",  "NR4A1", "GADD45G",  
    "SNAI2", "CREB5",  "HIST1H2BC", "HIST1H2BG"
]

df=df_main.iloc[:,:-1].copy()
marker_genes=marker
nr=4
nc=4
from anndata import AnnData
import scanpy as sc
import numpy as np

import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['figure.autolayout'] = True
import seaborn as sns

adata = AnnData(df.to_numpy())
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dfn = adata.to_df()
dfn.columns = df.columns
dfn['cell'] = df.index.values

dfn['umap1']= umap_coords[:,0]
dfn['umap2']= umap_coords[:,1]

fig, ax = plt.subplots(nr,nc) 
ax = ax.ravel()

for i,g in enumerate(marker_genes):
	if g in dfn.columns:
		print(g)
		val = np.array([x if x<3 else 3.0 for x in dfn[g]])
		sns.scatterplot(data=dfn, x='umap1', y='umap2', hue=val,s=1,palette="viridis",ax=ax[i],legend=False)

		# norm = plt.Normalize(val.min(), val.max())
		# sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
		# sm.set_array([])

		# cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
		# fig.colorbar(sm,ax=ax[i])
		# ax[i].axis('off')

		ax[i].set_title(g)
fig.savefig(wdir+cdir+'tg_umap_marker_genes_legend.png',dpi=600);plt.close()