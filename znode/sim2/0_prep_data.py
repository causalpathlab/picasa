
#### prep brca data

import anndata as an
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 

import scanpy as sc

wdir = '/data/sishir/data/sim2_multi/'

adata = an.read_h5ad(wdir+'sim2_multi_raw.h5ad')

dfl = adata.obs[['Cell','Batch','celltype']]
dfl.to_csv('znode/sim2/data/sim2_label.csv.gz',compression='gzip')

from picasa.util.hvgenes import select_hvgenes
hvgs = adata.var.index.values[select_hvgenes(adata.to_df().to_numpy(),gene_var_z=4.5)]
adata = adata[:,hvgs]

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=16, random_state=0).fit(adata.to_df().to_numpy())
# adata.obs['cluster'] = kmeans.labels_ 

# adata.obs.groupby(['Batch','cluster','celltype']).count().reset_index() 
 
# selected = adata.obs[(adata.obs.celltype == 'Group4') & (adata.obs.cluster ==0 )].index.values
# selected = np.concatenate([selected,  adata.obs[(adata.obs.celltype == 'Group1') & (adata.obs.cluster ==1 )].index.values])
# adata = adata[selected]

adata = adata[adata.obs.groupby(['Batch','celltype']).sample(n=400).index.values]

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

    adata_b.write('znode/sim2/data/sim2_'+str(batch)+'.h5ad',compression='gzip')