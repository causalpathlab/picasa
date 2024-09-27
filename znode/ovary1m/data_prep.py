
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt

########################################



adata = an.read_h5ad('ovary1m_full.h5ad')


selected_donors = [
'SPECTRUM-OV-007',
'SPECTRUM-OV-014',
'SPECTRUM-OV-022',
'SPECTRUM-OV-026',
'SPECTRUM-OV-036',
'SPECTRUM-OV-065',
'SPECTRUM-OV-081',
'SPECTRUM-OV-083',
'SPECTRUM-OV-107',
'SPECTRUM-OV-112'
]

adata = adata[adata.obs['donor_id'].isin(selected_donors)]


n_cells_per_donor = 1000
sampled_cells = (
    adata.obs.groupby(['donor_id','cell_type'], group_keys=False)
    .apply(lambda x: x.sample(min(len(x), n_cells_per_donor)))
)

##check
adata_sample= adata[sampled_cells.index, :]
df = adata_sample.obs.groupby(['donor_id','cell_type']).count().reset_index()

adata = adata_sample

remove_cols = [ x for x in adata.var['feature_name'].values if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]

keep_cols = [ x for x in adata.var['feature_name'].values if x not in remove_cols]

adata = adata[:,adata.var['feature_name'].isin(keep_cols)]


##remove genes whose sum is zero 
nonzero = (adata.to_df().sum()!=0).values
adata = adata[:,nonzero]


sc.pp.filter_genes(adata, min_cells=10)
### above drop 31k to 3k 


# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata,n_top_genes=2000)
hvgs = adata.var['highly_variable'].values
print(sum(hvgs))
adata = adata[:,hvgs]


df = adata.to_df()
df = df + (df.min().min() * -1)
df = df.div(df.sum(axis=1), axis=0) * 10000
df = df.round(0).astype(int)


import scipy.sparse as sp
import anndata
df_sparse = sp.csr_matrix(df.values)

adata_new = anndata.AnnData(X=df_sparse, obs=adata.obs.copy(), var=adata.var.copy())

adata_new.uns = adata.uns.copy()
adata_new.obsm = adata.obsm.copy()
adata_new.varm = adata.varm.copy()

adata = adata_new
adata.obs['batch'] = adata.obs['donor_id']
adata.obs['celltype'] = adata.obs['cell_type']
adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write('ovary_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs
dfl.to_csv('ovary_label.csv.gz',compression='gzip')


