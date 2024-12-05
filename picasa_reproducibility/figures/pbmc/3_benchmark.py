import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc
import harmonypy as hm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SAMPLE = 'pbmc'
WDIR = f'figures/{SAMPLE}/'
DATA_DIR = os.path.join(WDIR, 'data')
RESULTS_DIR = os.path.join(WDIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
PATTERN = f'{SAMPLE}_*.h5ad'
LABEL_FILE = os.path.join(DATA_DIR, f'{SAMPLE}_label.csv.gz')


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

def integrate_and_label_data(batch_map, label_file):
    combined_adata = an.concat(list(batch_map.values()), merge='unique', uns_merge='unique')
    combined_adata.X = combined_adata.X.astype(float)
    return combined_adata


def run_external_analysis(adata,df, method, save_path, batch_key='batch', group_key='celltype'):
    
    res = None
    
    if method == 'bbknn':
        import bbknn
        sc.pp.pca(adata)
        bbknn.bbknn(adata, batch_key=batch_key)        
    elif method == 'combat':
        sc.pp.combat(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        res = hm.compute_lisi(adata.obsm['X_pca'],adata.obs,[batch_key,group_key])
    elif method == 'harmony':
        sc.pp.pca(adata)
        sc.external.pp.harmony_integrate(adata, batch_key)
        sc.pp.neighbors(adata, use_rep='X_pca_harmony')
        res = hm.compute_lisi(adata.obsm['X_pca_harmony'],adata.obs,[batch_key,group_key])
    elif method == 'scanorama':
        sc.pp.pca(adata)
        sc.external.pp.scanorama_integrate(adata, batch_key)
        sc.pp.neighbors(adata, use_rep='X_scanorama')
        res = hm.compute_lisi(adata.obsm['X_scanorama'],adata.obs,[batch_key,group_key])
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        res = hm.compute_lisi(adata.obsm['X_pca'],adata.obs,[batch_key,group_key])
    
    sc.tl.umap(adata, min_dist=1.0)
    sc.pl.umap(adata, color=[batch_key, group_key])
    plt.savefig(save_path)
    plt.close()

    logging.info(f"UMAP saved to {save_path}")

    logging.info(f"LISI score {method}")
    
    if method != 'bbknn':    
        df_res = pd.DataFrame(res,columns=[method+'_'+batch_key,method+'_'+group_key])
        df = pd.concat([df,df_res],axis=1)

    
    return df

def run_picasa_analysis(df):
    picasa_adata = an.read_h5ad(os.path.join(RESULTS_DIR, 'picasa.h5ad'))
    ############ add metadata
    dfl= pd.read_csv(os.path.join(DATA_DIR,f'{SAMPLE}_label.csv.gz'))
    dfl.columns = ['index','cell','batch','celltype']
    dfl['cell'] = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]
    dfl = dfl[['index','cell','celltype']]
    
    picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')
    picasa_res = hm.compute_lisi(picasa_adata.obsm['common'],picasa_adata.obs,['batch','celltype'])
    df_picasa_res = pd.DataFrame(picasa_res,columns=['picasa_batch','picasa_celltype'])
    df = pd.concat([df,df_picasa_res],axis=1)
    return df

    

if __name__ == "__main__":
    logging.info("Starting batch integration pipeline")
    
    batch_map = load_batches(DATA_DIR, PATTERN)
    adata_main = integrate_and_label_data(batch_map, LABEL_FILE)
        
    methods = {
        'uncorrected': os.path.join(RESULTS_DIR, 'scanpy_umap.png'),
        'bbknn': os.path.join(RESULTS_DIR, 'scanpy_umap_bbknn.png'),
        'combat': os.path.join(RESULTS_DIR, 'scanpy_umap_combat.png'),
        'harmony': os.path.join(RESULTS_DIR, 'scanpy_umap_harmony.png'),
        'scanorama': os.path.join(RESULTS_DIR, 'scanpy_umap_scanorama.png'),
    }
    
    df = pd.DataFrame()
    
    for method, save_path in methods.items():
        logging.info(f"Running {method.upper()}")
        adata = adata_main.copy()
        df = run_external_analysis(adata,df, method, save_path)
    
    df = run_picasa_analysis(df)
    df = df.melt()
    df.columns = ['batch','lisi']
    df['method'] = [x.split('_')[0] for x in df['batch']]
    df['type'] = [x.split('_')[1] for x in df['batch']]
    
    

    df.to_csv(os.path.join(RESULTS_DIR, 'benchmark.csv.gz'),compression='gzip',index=False)
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'benchmark.csv.gz'))
    logging.info("All analyses completed.")

    df = df[df['type']=='batch']
    
    sn.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))  
    sn.boxplot(data=df, x="lisi", y="method", palette="Set2")
    sn.stripplot(data=df, x="lisi", y="method", color="grey", size=1, alpha=0.3, jitter=True)
    plt.xlabel("LISI Score", fontsize=12)
    plt.ylabel("Integration Method", fontsize=12)
    plt.title("Comparison of LISI Scores Across Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'benchmark_lisi.png'))
    plt.close()
    
