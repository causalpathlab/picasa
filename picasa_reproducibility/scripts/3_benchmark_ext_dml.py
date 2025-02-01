import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc

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


import scDML
from scDML import scDMLModel
## fullrun
save_dir=WDIR+SAMPLE+"/results/"

ncluster = adata_combined.obs[constants.GROUP].nunique()

adata_combined.obs['BATCH'] = adata_combined.obs['batch']



scdml=scDMLModel(save_dir=save_dir)
adata=scdml.preprocess(adata_combined,cluster_method="louvain",resolution=3.0)



scdml.integrate(adata,batch_key="BATCH",ncluster_list=[ncluster],
expect_num_cluster=ncluster,merge_rule="rule1",num_epochs=2)

# scdml.integrate(adata,batch_key="BATCH",ncluster_list=[ncluster],
# expect_num_cluster=ncluster,merge_rule="rule2")

adata.obs['batch'] = adata.obs['BATCH']



sc.pp.neighbors(adata, use_rep="X_emb")
sc.tl.umap(adata)


sc.pl.umap(adata, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_dml_umap_'+constants.BATCH+'.png'))
plt.close()

sc.pl.umap(adata, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_dml_umap_'+constants.GROUP+'.png'))
plt.close()


pd.DataFrame(adata.obsm['X_emb'],index=adata_combined.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_dml.csv.gz'),compression='gzip')
