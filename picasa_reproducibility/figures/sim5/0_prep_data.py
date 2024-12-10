
import pandas as pd
from scipy.sparse import csr_matrix
import anndata as an


adata = an.read_h5ad('PBMC.merged.h5ad')

adata.obs['batch'] = ['batch_'+str(x) for x in adata.obs['batch'].values]
adata.obs['celltype'] = adata.obs['Cell type'].values

print(adata.obs.celltype.value_counts())
print(adata.obs.batch.value_counts())



wdir = ''

batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write(wdir+'pbmc_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.to_csv(wdir+'pbmc_label.csv.gz',compression='gzip')


