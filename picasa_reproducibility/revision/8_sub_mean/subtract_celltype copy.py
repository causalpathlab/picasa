import os
import glob
import logging
import matplotlib.pyplot as plt
import anndata as an
import pandas as pd
import scanpy as sc
import numpy as np


import sys 

import constants 

SAMPLE = "sim1"
WDIR = '/Users/sishirsubedi/Documents/projects/picasa/revision/'

DATA_DIR = os.path.join(WDIR, 'data')
RESULTS_DIR = os.path.join(WDIR,'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
PATTERN = f'{SAMPLE}_*.h5ad'


def load_batches(data_dir, pattern, max_batches=25):
    batch_files = glob.glob(os.path.join(data_dir, pattern))
    batch_map = {}
    for i, file in enumerate(batch_files):
        if i >= max_batches:
            break
        batch_name = os.path.basename(file).replace('.h5ad', '').replace(f'{SAMPLE}_', '')
        logging.info(f"Loading {batch_name}")
        batch_map[batch_name] = an.read_h5ad(file)
    return batch_map

def integrate_data(batch_map):
    combined_adata = an.concat(list(batch_map.values()), merge='unique', uns_merge='unique')
    combined_adata.X = combined_adata.X.astype(float)
    return combined_adata



batch_map = load_batches(DATA_DIR, PATTERN)  
adata_main = integrate_data(batch_map)
adata = adata_main.copy()
        
sc.pp.combat(adata,key='celltype')
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

X = adata.X
clusters = adata.obs["leiden"].values

X_residual = np.zeros_like(X)

for cluster in np.unique(clusters):
    idx = clusters == cluster
    cluster_mean = X[idx].mean(axis=0)
    X_residual[idx] = X[idx] - cluster_mean

adata_residual = adata.copy()
adata_residual.X = X_residual

sc.pp.pca(adata_residual)
sc.pp.neighbors(adata_residual)
sc.tl.leiden(adata_residual)
sc.tl.umap(adata_residual)

sc.pl.umap(
    adata_residual,
    color=["leiden", "batch", "celltype"]
)

pd.crosstab(
    adata_residual.obs["leiden"],
    adata_residual.obs["batch"],
    normalize="index"
)