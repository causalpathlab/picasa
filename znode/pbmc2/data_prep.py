
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


import anndata as an 

adata_main = an.read_h5ad('pbmc.h5ad')

psel = pd.DataFrame(adata_main.obs['platform'].isin(["10x3'v3", "10x3'v2"]))
psel = psel[psel['platform']==True].index.values


## select platform
df = adata_main.to_df()
df = df.loc[psel,:]

## sample cells
dfl = pd.merge(df['MIR1302-2HG'],adata_main.obs,left_index=True, right_index=True, how='left')
dfl = dfl.groupby('platform').sample(n=6000)
dfl['ct'].value_counts()
df = df.loc[dfl.index.values,:]


from picasa.util.hvgenes import select_hvgenes

hvgs = df.columns.values[select_hvgenes(df.to_numpy(),gene_var_z=2)]
print(len(hvgs))

df = df.loc[:,hvgs]




wdir = 'znode/pbmc2/'



smat = csr_matrix(df.to_numpy())
adata = an.AnnData(X=smat)
adata.var_names = df.columns.values
adata.obs_names = df.index.values
adata.obs['batch'] = [ x.replace("10x3'","") for x in dfl['platform'].values]
adata.obs['celltype'] = dfl['ct'].values

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
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write(wdir+'pbmc2_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.columns = ['cell','batch','celltype']
dfl.to_csv(wdir+'pbmc2_label.csv.gz',compression='gzip')


