
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc
import h5py as hf

import sys
sys.path.append("/home/BCCRC.CA/ssubedi/projects/experiments/picasa/")


def get_adata(fn,batch):
    f = hf.File(fn,'r')

    mtx_indptr = f['matrix']['indptr']
    mtx_indices = f['matrix']['indices']
    mtx_data = f['matrix']['data']
    barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
    features = [x.decode('utf-8') for x in f['matrix']['features']['id']]

    smat = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(barcodes),len(features)))

    adata = an.AnnData(X=smat)
    adata.var_names = features
    adata.obs_names = barcodes
    adata.obs['batch'] = [ batch for x in range(adata.shape[0])]

    adata.obs['celltype'] = [ '-'.join(x.split('_')[1:]) for x in adata.obs.index.values]

    return adata

wdir = 'znode/sim3/data/'
ad1 = get_adata(wdir+'sim3_b1.h5','batch1')
ad2 = get_adata(wdir+'sim3_b2.h5','batch2')
ad3 = get_adata(wdir+'sim3_b3.h5','batch3')

adata = an.concat([ad1, ad2], join='outer')
adata = an.concat([adata, ad3], join='outer')


adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()

# from picasa.util.hvgenes import select_hvgenes

# hvgs = adata.var.index.values[select_hvgenes(adata.to_df().to_numpy(),gene_var_z=3.0)]
# len(hvgs)
# adata = adata[:,hvgs]

batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df = adata_c.to_df()

    smat = csr_matrix(df.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df.columns.values
    adata_b.obs_names = df.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write(wdir+'sim3_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.columns = ['cell','batch','celltype']
dfl.to_csv(wdir+'sim3_label.csv.gz',compression='gzip')
