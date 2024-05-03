
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc
import matplotlib.pylab as plt

wdir = 'znode/membryo/data/'
adata_sc = an.read_h5ad(wdir+'adata_rna.h5ad')
adata_sp = an.read_h5ad(wdir+'adata_seqfish_40.h5ad')

adata_sc = adata_sc[:,adata_sp.var.index.values]

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(adata_sp.obs['X'],adata_sp.obs['Y'])]


common_celltype = np.intersect1d(adata_sc.obs.cell_type.unique(),adata_sp.obs.cell_type.unique())
adata_sp = adata_sp[adata_sp.obs['cell_type'].isin(common_celltype)]
adata_sc = adata_sc[adata_sc.obs['cell_type'].isin(common_celltype)]

adata_sp.write(wdir+'membryo_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'membryo_sc.h5ad',compression='gzip')