
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc
import matplotlib.pylab as plt

wdir = 'znode/brca/data/'
df_sc = pd.read_csv(wdir+'brca_scRNA_GEP_subset.txt',sep='\t')
df_sp = pd.read_csv(wdir+'brca_STdata_GEP.txt',sep='\t')

df_sc.set_index('Gene',inplace=True)
df_sp.set_index('V1',inplace=True)

## filter genes based on total gene count
df_sp = df_sp[df_sp.sum(1)>100]
df_sc = df_sc[df_sc.sum(1)>100]

common_genes = np.intersect1d(df_sc.index.values,df_sp.index.values)

df_sc = df_sc.loc[common_genes,:]
df_sp = df_sp.loc[common_genes,:]

df_sc = df_sc.T
df_sp = df_sp.T

from picasa.util.hvgenes import select_hvgenes

sp_hvgs = df_sp.columns.values[select_hvgenes(df_sp.to_numpy(),gene_var_z=2)]
sc_hvgs = df_sc.columns.values[select_hvgenes(df_sc.to_numpy(),gene_var_z=2)]

common_hvgs = np.intersect1d(sp_hvgs,sc_hvgs)

df_sc = df_sc.loc[:,common_hvgs]
df_sp = df_sp.loc[:,common_hvgs]

smat = csr_matrix(df_sc.to_numpy())
adata_sc = an.AnnData(X=smat)
adata_sc.var_names = df_sc.columns.values
adata_sc.obs_names = df_sc.index.values

smatsp = csr_matrix(df_sp.to_numpy())
adata_sp = an.AnnData(X=smatsp)
adata_sp.var_names = df_sp.columns.values
adata_sp.obs_names = df_sp.index.values





dfspl = pd.read_csv(wdir+'brca_STdata_coordinates.txt',sep='\t')
dfspl.set_index('SpotID',inplace=True)
dfspl.columns = ['x','y']

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(dfspl['x'],dfspl['y'])]



adata_sp.write(wdir+'brca_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'brca_sc.h5ad',compression='gzip')