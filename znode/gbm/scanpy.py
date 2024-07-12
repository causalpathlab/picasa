<<<<<<< HEAD
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
    adata.obs['batch'] = [batch_name] * n_obs
    # Here we use a placeholder for celltype, adjust based on your actual data
    adata.obs['celltype'] = ['unknown'] * n_obs

# Concatenate all AnnData objects
combined_adata = an.concat([adata for adata in batch_map.values()], merge='unique', uns_merge='unique')


import scanpy as sc
import matplotlib.pyplot as plt
adata = combined_adata
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
plt.savefig(wdir+'scanpy.png')


dfl = pd.DataFrame(adata.obs['leiden'])
dfl.reset_index(inplace=True)
dfl.columns = ['cell','leiden']
dfl.to_csv(wdir+'data/gbm_label_leiden.csv.gz',index=False,compression='gzip')
=======
# import sys 
# sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

# import matplotlib.pylab as plt
# import seaborn as sns
# import anndata as an
# import pandas as pd
# import numpy as np
# import picasa
# import torch
# import logging


# import glob
# import os

# sample = 'gbm'
# wdir = 'znode/gbm/'

# directory = wdir+'/data'
# pattern = 'gbm_*.h5ad'

# file_paths = glob.glob(os.path.join(directory, pattern))
# file_names = [os.path.basename(file_path) for file_path in file_paths]

# batch_map = {}
# batch_count = 0
# for file_name in file_names:
# 	print(file_name)
# 	batch_map[file_name.replace('.h5ad','').replace('gbm_','')] = an.read_h5ad(wdir+'data/'+file_name)
# 	batch_count += 1
# 	if batch_count >25:
# 		break


# for batch_name, adata in batch_map.items():
#     n_obs = adata.shape[0]
#     adata.obs['batch'] = batch_name
#     adata.obs['celltype'] = 'unknown'

# combined_adata = an.concat([adata for adata in batch_map.values()], merge='unique', uns_merge='unique')


# import scanpy as sc
# import matplotlib.pyplot as plt
# adata = combined_adata

# sc.pp.filter_cells(adata, min_genes=25)
# sc.pp.filter_genes(adata, min_cells=2)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# adata = adata[:, adata.var.highly_variable]
# sc.tl.pca(adata,random_state=42)


# sc.pp.neighbors(adata)
# sc.tl.leiden(adata,resolution=0.1)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=["leiden"])
# plt.savefig(wdir+'scanpy_leiden.png')

# sc.pl.umap(adata, color=["batch"])
# plt.savefig(wdir+'scanpy_batch.png')


# dfl = pd.DataFrame(adata.obs['leiden'])
# dfl.reset_index(inplace=True)
# dfl.columns = ['cell','leiden']
# dfl.to_csv(wdir+'data/gbm_label_leiden.csv.gz',index=False,compression='gzip')


# adata_uncorrected = adata.copy()
# import bbknn
# bbknn.bbknn(adata, batch_key='batch')

# sc.tl.leiden(adata,resolution=0.1)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=["leiden","batch"])
# plt.savefig(wdir+'scanpy_leiden_corr.png')

>>>>>>> with_weighted_rare_celltype
