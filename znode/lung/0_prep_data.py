
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc

wdir = '/data/sishir/data/Lung_two_species/'

adata = an.read_h5ad(wdir+'Lung_two_species_raw.h5ad')


df = adata.to_df()
from picasa.util.hvgenes import select_hvgenes

hvgs = df.columns.values[select_hvgenes(df.to_numpy(),gene_var_z=5)]
print(len(hvgs))


marker = ["CD3G", "CD8A", "SOX9", "ACTA2", "SCGB3A2", "GUCY1A1", "NKG7", "S100A8", "GNGT2", 
         "CD14", "MSLN", "MS4A2", "C1QB",  "GATA3", "DCN", "ITGA8", "VCAM1", 
         "CLEC14A", "MMRN1", "PRX", "CD7", "ITGAE", "CCL17", "CD86", "CCDC113", "TOP2A", 
         "KRT5", "JCHAIN", "CD79B", "BCL11A", "SFTPB", "AGER"]

hvgs = np.concatenate((hvgs,np.array(marker)))
print(len(hvgs))

hvgs = np.unique(hvgs)
len(hvgs)

adata = adata[:,hvgs]

adata.obs['batch'] = adata.obs['BATCH'].values
adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

ndir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/znode/lung/data/'
for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df = adata_c.to_df()

    smat = csr_matrix(df.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df.columns.values
    adata_b.obs_names = df.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write(ndir+'lung_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl = dfl.loc[:,['index','batch','celltype']]
dfl.columns = ['cell','batch','celltype']
dfl.to_csv(ndir+'lung_label.csv.gz',compression='gzip')

