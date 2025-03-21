import os
import glob
import logging
import matplotlib.pyplot as plt
import anndata as an
import pandas as pd
import scanpy as sc
import scvi

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1] 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'

DATA_DIR = os.path.join(WDIR, SAMPLE, 'model_data')
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


picasa_adata = an.read_h5ad(WDIR+SAMPLE+'/model_results/picasa.h5ad')
n_latent=picasa_adata.obsm['common'].shape[1]


batch_map = load_batches(DATA_DIR, PATTERN)

adata_combined = sc.concat(batch_map, join="outer", label="batch")

scvi.model.SCVI.setup_anndata(adata_combined, batch_key="batch") 
model = scvi.model.SCVI(adata_combined,n_latent=n_latent)
model.train()

adata_combined.obsm["X_scVI"] = model.get_latent_representation()
pd.DataFrame(adata_combined.obsm['X_scVI'],index=adata_combined.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_scvi.csv.gz'),compression='gzip')


sc.pp.neighbors(adata_combined, use_rep="X_scVI")
sc.tl.umap(adata_combined)

sc.pl.umap(adata_combined, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_scvi_umap_'+constants.BATCH+'.png'))
plt.close()

sc.pl.umap(adata_combined, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_scvi_umap_'+constants.GROUP+'.png'))
plt.close()
