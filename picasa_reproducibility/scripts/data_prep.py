
import anndata as an

import pandas as pd
import numpy as np
import constants

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt



def download_file(url,dest_file):
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully as '{dest_file}'")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def prep_pbmc_data():
            
    import h5py as hf
    
    url_3p = "https://cf.10xgenomics.com/samples/cell-exp/4.0.0/Parent_NGSC3_DI_PBMC/Parent_NGSC3_DI_PBMC_filtered_feature_bc_matrix.h5"
    dest_file_3p = "3p_pbmc10k_filt.h5"
        
    url_5p = "https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5"
    dest_file_5p = "5p_pbmc10k_filt.h5"

    download_file(url_3p,dest_file_3p)
    download_file(url_5p,dest_file_5p)

    data_3p = hf.File(dest_file_3p)


def prep_pbmc3k(meta_file,data_file):
    

    
    df = pd.read_csv(data_file,header=0)
    df = df.T
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:]
    df = df.astype(int)

    dfl = pd.read_csv(meta_file,header=0)
    dfl.columns = ['cell','celltype','batch']

# smat = csr_matrix(df.to_numpy())
# adata = an.AnnData(X=smat)
# adata.var_names = df.columns.values
# adata.obs_names = df.index.values
# adata.obs['batch'] = [ x.split('_')[0] for x in df.index.values]

def prep_sim5_data():
    file_path = '/data/sishir/data/batch_correction/sim1_multi/sim1_multi_raw.h5ad'
    adata = an.read(file_path)
    
    adata.obs = adata.obs.iloc[:,:3]
    column_map = {
        'Cell'  : constants.SAMPLE,
        'Batch' : constants.BATCH,
        'Group' : constants.GROUP,
        }
    adata.obs.rename(columns=column_map,inplace=True)

    return adata

def prep_sim6_data():
    file_path = '/data/sishir/data/batch_correction/sim1_multi/sim1_multi_raw.h5ad'
    adata = an.read(file_path)
    
    adata.obs = adata.obs.iloc[:,:3]
    column_map = {
        'Cell'  : constants.SAMPLE,
        'Batch' : constants.BATCH,
        'Group' : constants.GROUP,
        }
    adata.obs.rename(columns=column_map,inplace=True)

    adata = adata[ (adata.obs[constants.BATCH] != 'Batch5') \
         & (adata.obs[constants.BATCH] != 'Batch6') \
         & (adata.obs[constants.GROUP] != 'Group7') \
         ]
    return adata

def prep_pancreas_data():
    file_path = '/data/sishir/data/batch_correction/pancreas/spancreas_raw.h5ad'
    adata = an.read(file_path)
    
    adata.var.set_index('_index',inplace=True)
    
    adata.obs['Cell'] = adata.obs.index.values
    
    adata.obs = adata.obs[['Cell','BATCH','celltype']]
    column_map = {
        'Cell'  : constants.SAMPLE,
        'BATCH' : constants.BATCH,
        'celltype' : constants.GROUP,
        }
    adata.obs.rename(columns=column_map,inplace=True)

    return adata


def qc():
    import scanpy as sc  
      
    remove_cols = [ x for x in adata.var.index.values if \
        x.startswith('MT-') \
        or x.startswith('RPL') \
        or x.startswith('RPS') \
        or x.startswith('RP1') \
        or x.startswith('MRP')
    ]
    
    keep_cols = [ x for x in adata.var.index.values if x  not in remove_cols]
    adata = adata[:,keep_cols]


    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata,n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]


def generate_batch_data(adata,sample_name):
    batch_keys = list(adata.obs[constants.BATCH].unique())

    for batch in batch_keys:
        adata_c = adata[adata.obs[constants.BATCH].isin([batch])]
        df = adata_c.to_df()

        smat = csr_matrix(df.to_numpy())
        adata_b = an.AnnData(X=smat)
        adata_b.var_names = df.columns.values
        adata_b.obs_names = df.index.values
        adata_b.obs[constants.BATCH] = adata_c.obs[constants.BATCH].values
        adata_b.obs[constants.GROUP] = adata_c.obs[constants.GROUP].values

        adata_b.write(sample_name+'_'+str(batch)+'.h5ad',compression='gzip')


generate_batch_data(adata,'pancreas')

