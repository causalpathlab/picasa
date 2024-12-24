import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc
import harmonypy as hm

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

def integrate_data(batch_map):
    combined_adata = an.concat(list(batch_map.values()), merge='unique', uns_merge='unique')
    combined_adata.X = combined_adata.X.astype(float)
    return combined_adata


def run_scanpy_external_analysis(adata, method, save_path, batch_key=constants.BATCH, group_key=constants.GROUP):
        
    if method == 'bbknn':
        import bbknn
        sc.pp.pca(adata)
        bbknn.bbknn(adata, batch_key=batch_key)        
    elif method == 'combat':
        sc.pp.combat(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        pd.DataFrame(adata.obsm['X_pca'],index=adata.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_combat.csv.gz'),compression='gzip')
    elif method == 'harmony':
        sc.pp.pca(adata)
        sc.external.pp.harmony_integrate(adata, batch_key)
        sc.pp.neighbors(adata, use_rep='X_pca_harmony')
        pd.DataFrame(adata.obsm['X_pca_harmony'],index=adata.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_harmony.csv.gz'),compression='gzip')
    elif method == 'scanorama':
        sc.pp.pca(adata)
        sc.external.pp.scanorama_integrate(adata, batch_key)
        sc.pp.neighbors(adata, use_rep='X_scanorama')
        pd.DataFrame(adata.obsm['X_scanorama'],index=adata.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_scanorama.csv.gz'),compression='gzip')
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        res = hm.compute_lisi(adata.obsm['X_pca'],adata.obs,[batch_key,group_key])
        pd.DataFrame(adata.obsm['X_pca'],index=adata.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_pca.csv.gz'),compression='gzip')
    
    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color=[batch_key],legend_loc=None)
    plt.savefig(os.path.join(save_path+'_'+batch_key+'.png'))
    plt.close()

    sc.pl.umap(adata, color=[group_key],legend_loc=None)
    
    plt.savefig(os.path.join(save_path+'_'+group_key+'.png'))
    plt.close()

    logging.info(f"UMAP saved to {save_path}")

    logging.info(f"LISI score {method}")


    
if __name__ == "__main__":
    logging.info("Starting batch integration pipeline")
    
    batch_map = load_batches(DATA_DIR, PATTERN)
    adata_main = integrate_data(batch_map)
        
    methods = {
        'uncorrected': os.path.join(RESULTS_DIR, 'scanpy_pca_umap'),
        'bbknn': os.path.join(RESULTS_DIR, 'scanpy_bbknn_umap'),
        'combat': os.path.join(RESULTS_DIR, 'scanpy_combat_umap'),
        'harmony': os.path.join(RESULTS_DIR, 'scanpy_harmony_umap'),
        'scanorama': os.path.join(RESULTS_DIR, 'scanpy_scanorama_umap'),
    }
        
    for method, save_path in methods.items():
        logging.info(f"Running {method.upper()}")
        adata = adata_main.copy()
        run_scanpy_external_analysis(adata, method, save_path)
    