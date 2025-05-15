import os
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import anndata as an
import pandas as pd
import scanpy as sc
import numpy as np

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import pyliger
import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1] 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
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


    

batch_map = load_batches(DATA_DIR, PATTERN)

adata_combined = sc.concat(batch_map, join="outer", label="batch")

for batch, adata in batch_map.items():
	adata.var.rename(columns={0:'gene'},inplace=True) 
	adata.obs.rename(columns={0:'cell'},inplace=True) 
	adata.obs.index.name = 'cell'
	adata.var.index.name = 'gene'

adata_list = []
for batch, adata in batch_map.items():
    adata.uns['sample_name'] = batch
    adata_list.append(adata)

picasa_adata = an.read_h5ad(WDIR+SAMPLE+'/results/picasa.h5ad')
K=picasa_adata.obsm['common'].shape[1]


ifnb_liger = pyliger.create_liger(adata_list,remove_missing=False)
 
pyliger.normalize(ifnb_liger) 
pyliger.select_genes(ifnb_liger)

gene_use = 2000
result = []
for i in range(len(ifnb_liger.adata_list)):
    ifnb_liger.adata_list[i].uns['var_gene_idx'] = ifnb_liger.adata_list[i].var['norm_var'].sort_values(ascending=False).index.values[:gene_use]
    result.append(ifnb_liger.adata_list[i].uns['var_gene_idx'])

from functools import reduce

result = reduce(np.union1d, np.array(result))
ifnb_liger.var_genes = result
  
pyliger.scale_not_center(ifnb_liger,remove_missing=False)

pyliger.optimize_ALS(ifnb_liger, k = 15)
pyliger.quantile_norm(ifnb_liger)

h_norm = np.vstack([adata.obsm["H_norm"] for adata in ifnb_liger.adata_list])


adata_combined.obsm['X_liger'] = h_norm

pd.DataFrame(adata_combined.obsm['X_liger'],index=adata_combined.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_liger.csv.gz'),compression='gzip')


fig, ax = plt.subplots(figsize=(6, 6))


sc.pp.neighbors(adata_combined, use_rep="X_liger")
sc.tl.umap(adata_combined)


sc.pl.umap(adata_combined, color=[constants.BATCH],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_liger_umap_'+constants.BATCH+'.png'))
plt.close()

sc.pl.umap(adata_combined, color=[constants.GROUP],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_liger_umap_'+constants.GROUP+'.png'))
plt.close()

