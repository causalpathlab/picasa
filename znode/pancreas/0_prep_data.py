
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc

wdir = '/data/sishir/data/batch_correction/pancreas/'

adata = an.read_h5ad(wdir+'pancreas_raw.h5ad')

adata.var.set_index('_index',inplace=True)

df = adata.to_df()

from picasa.util.hvgenes import select_hvgenes_scanpy

hvgs = df.columns.values[select_hvgenes_scanpy(df.to_numpy(),n_genes=2000)]

print(len(hvgs))


adata = adata[:,hvgs]


adata.obs['batch'] = adata.obs['BATCH'].values
adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


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

    adata_b.write('pancreas_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl = dfl.loc[:,['index','batch','celltype']]
dfl.columns = ['cell','batch','celltype']
dfl.to_csv('pancreas_label.csv.gz',compression='gzip')

