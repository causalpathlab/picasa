
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc

wdir = '/data/sishir/data/sim1_multi/'

adata = an.read_h5ad(wdir+'sim1_multi_raw.h5ad')




from picasa.util.hvgenes import select_hvgenes
hvgs = adata.var.index.values[select_hvgenes(adata.to_df().to_numpy(),gene_var_z=2.5)]
adata = adata[:,hvgs]

batch_keys = list(adata.obs['Batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['Batch'].isin([batch])]
    df = adata_c.to_df()

    smat = csr_matrix(df.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df.columns.values
    adata_b.obs_names = df.index.values
    adata_b.obs['batch'] = adata_c.obs['Batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write('znode/sim1/data/sim1_'+str(batch)+'.h5ad',compression='gzip')