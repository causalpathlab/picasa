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

sample = 'gbm'
wdir = 'znode/gbm/'

directory = wdir+'/data'
pattern = 'gbm_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('gbm_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >25:
		break


for batch_name, adata in batch_map.items():
    n_obs = adata.shape[0]
    adata.obs['batch'] = batch_name
    adata.obs['celltype'] = 'unknown'

combined_adata = an.concat([adata for adata in batch_map.values()], merge='unique', uns_merge='unique')
adata = combined_adata


df_umap = pd.read_csv(wdir+'results/df_umap.csv.gz')
sel_cells = df_umap[df_umap['cluster']==0]['cell'].values


# adata = adata[sel_cells]



import scanpy as sc


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dfn = adata.to_df()
dfn.columns = [x.split('_')[1] for x in adata.var.index.values]
dfn['cell'] = adata.obs.index.values

dfn = pd.merge(dfn,df_umap,on='cell',how='left')

# from analysis import _supp


marker_genes = ["TOP2A", "AURKB", "FOXM1", "TYMS", "USP1", "EZH2"]
plot_marker('1',dfn,marker_genes)
marker_genes = ["APOD", "OLIG2","STMN1", "DCX", "SOX11", "TNC"]
plot_marker('2',dfn,marker_genes)
marker_genes = ["CD44", "S100A10", "VIM", "HLA-A","APOE", "HSPA1B"]
plot_marker('3',dfn,marker_genes)
marker_genes = ["PDGFRA", "SOX2", "DCX", "NG2","GFAP", "S100B"]
plot_marker('4',dfn,marker_genes)

    
    

import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.autolayout'] = True
import colorcet as cc
import seaborn as sns
import numpy as np

def plot_marker(fn,df,marker_genes):

    fig, ax = plt.subplots(2,3) 
    ax = ax.ravel()

    for i,g in enumerate(marker_genes):
        if g in df.columns:
            print(g)
            val = np.array([x if x<3 else 3.0 for x in df[g]])
            sns.scatterplot(data=df, x='umap1', y='umap2', hue=val,s=.1,palette="viridis",ax=ax[i],legend=False)

            norm = plt.Normalize(val.min(), val.max())
            sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
            sm.set_array([])

            # cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
            fig.colorbar(sm,ax=ax[i])
            ax[i].axis('off')

            ax[i].set_title(g)
    fig.savefig(fn+'.png')
                
                
import scanpy as sc
import matplotlib.pyplot as plt

sc.pp.filter_cells(adata, min_genes=25)
sc.pp.filter_genes(adata, min_cells=2)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var.highly_variable]
sc.tl.pca(adata,random_state=42)


sc.pp.neighbors(adata)
sc.tl.leiden(adata,resolution=0.1)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["leiden"])
plt.savefig(wdir+'scanpy_leiden_c.png')

sc.pl.umap(adata, color=["batch"])
plt.savefig(wdir+'scanpy_batch_c.png')


dfl = pd.DataFrame(adata.obs['leiden'])
dfl.reset_index(inplace=True)
dfl.columns = ['cell','leiden']
dfl.to_csv(wdir+'data/gbm_label_leiden.csv.gz',index=False,compression='gzip')


adata_uncorrected = adata.copy()
import bbknn
bbknn.bbknn(adata, batch_key='batch')

sc.tl.leiden(adata,resolution=0.1)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["leiden","batch"])
plt.savefig(wdir+'scanpy_leiden_corr.png')

