import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc
import scvi

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1] 
WDIR = sys.argv[2]


DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'results')
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

import anndata as ad
import scanpy as sc
import cellanova as cnova

adata_prep = cnova.model.preprocess_data(adata_combined, integrate_key=constants.BATCH)


## model fitting
adata_prep= cnova.model.calc_ME(adata_prep, integrate_key=constants.BATCH)

control_dict = {
    'g1': adata_prep.obs.batch.unique(),
}
adata_prep = cnova.model.calc_BE(adata_prep, 
                                 integrate_key=constants.BATCH, 
                                 control_dict=control_dict)

adata_prep.layers['batch_effect'] = adata_prep.layers['scale'] - adata_prep.layers['corrected']


sc.pp.pca(adata_prep,layer="main_effect")
sc.pp.neighbors(adata_prep, use_rep="X_pca")
sc.tl.umap(adata_prep)
sc.pl.umap(adata_prep, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_cellanova_umap_'+constants.BATCH+'.png'))
plt.close()

sc.pl.umap(adata_prep, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_cellanova_umap_'+constants.GROUP+'.png'))
plt.close()


pd.DataFrame(adata_prep.obsm['X_pca'],index=adata_combined.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_cellanova.csv.gz'),compression='gzip')


sc.pp.pca(adata_prep,layer="batch_effect")
sc.pp.neighbors(adata_prep, use_rep="X_pca")
sc.tl.umap(adata_prep)
sc.pl.umap(adata_prep, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_cellanova_umap_'+constants.BATCH+'_unique.png'))
plt.close()

sc.pl.umap(adata_prep, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_cellanova_umap_'+constants.GROUP+'_unique.png'))
plt.close()
