

import pandas as pd
from scipy.sparse import csr_matrix
import h5py as hf
import anndata as an


wdir = "data/"

df= pd.read_csv(wdir+'sim_sc_count.csv.gz').T
dfl= pd.read_csv(wdir+'sim_sc_label.csv.gz')
df.columns  = df.iloc[0,:]
df = df.iloc[1:,:]
df = df.astype(int)

smat = csr_matrix(df.to_numpy())
adata = an.AnnData(X=smat)
adata.var_names = df.columns.values 
adata.obs_names = df.index.values


adata.obs['batch'] = [x.replace('10x Chromium (','').replace(')','') for x in dfl['batch'].values]
adata.obs['celltype'] = dfl['cell_type'].values

print(adata.obs.celltype.value_counts())
print(adata.obs.batch.value_counts())


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write(wdir+'sim8_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.to_csv(wdir+'sim_sc_label.csv.gz',compression='gzip')