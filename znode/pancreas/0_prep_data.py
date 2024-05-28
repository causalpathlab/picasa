
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc

wdir = '/data/sishir/data/pancreas/'

adata = an.read_h5ad(wdir+'pancreas_raw.h5ad')

adata_scm = adata[adata.obs['BATCH'].isin(['indrop3'])]
df_sc = adata_scm.to_df()

adata_spm = adata[adata.obs['BATCH'].isin(['smartseq2'])]
df_sp = adata_spm.to_df()


from picasa.util.hvgenes import select_hvgenes

sp_hvgs = df_sp.columns.values[select_hvgenes(df_sp.to_numpy(),gene_var_z=3.5)]
sc_hvgs = df_sc.columns.values[select_hvgenes(df_sc.to_numpy(),gene_var_z=3.5)]

common_hvgs = np.intersect1d(sp_hvgs,sc_hvgs)

print(len(common_hvgs))

df_sc = df_sc.loc[:,common_hvgs]
df_sp = df_sp.loc[:,common_hvgs]

smat = csr_matrix(df_sc.to_numpy())
adata_sc = an.AnnData(X=smat)
adata_sc.var_names = df_sc.columns.values
adata_sc.obs_names = df_sc.index.values
adata_sc.obs['batch'] = adata_scm.obs['BATCH'].values
adata_sc.obs['celltype'] = adata_scm.obs['celltype'].values



smatsp = csr_matrix(df_sp.to_numpy())
adata_sp = an.AnnData(X=smatsp)
adata_sp.var_names = df_sp.columns.values
adata_sp.obs_names = df_sp.index.values
adata_sp.obs['batch'] = adata_spm.obs['BATCH'].values
adata_sp.obs['celltype'] = adata_spm.obs['celltype'].values



adata_sp.write('pancreas_sp.h5ad',compression='gzip')
adata_sc.write('pancreas_sc.h5ad',compression='gzip')