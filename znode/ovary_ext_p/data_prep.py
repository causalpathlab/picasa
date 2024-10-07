
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


## get common genes from train data

adata = an.read_h5ad('../data/ovary_EOC3.h5ad')
common_genes = adata.var.index.values
########### read raw from web and generate main adata


## in /data/sishir/ovary_ext/

files = [
    'GSM3729170_P1_dge',
    'GSM3729171_P2_dge',
    'GSM3729172_P3_dge',
    'GSM3729173_P4_dge'
    ]

df = pd.DataFrame() 
for f in files:
    dfc =  pd.read_csv(f+'.txt.gz',sep='\t')
    dfc = dfc.T
    dfc.columns = dfc.iloc[0,:]
    dfc = dfc.iloc[1:,:]
    dfc = dfc[common_genes]
    dfc.index = [f.split('_')[1] + '@' + x for x in dfc.index.values]
    df = pd.concat([df,dfc],axis=0)
    print(df.shape)

df = df.astype(int)

import scipy.sparse as sp
import anndata
df_sparse = sp.csr_matrix(df.values)

adata = anndata.AnnData(X=df_sparse, obs=pd.DataFrame(df.index.values), var=pd.DataFrame(df.columns.values))

adata.obs.index = adata.obs[0].values
adata.var.index = adata.var[0].values
adata.obs['batch'] = [x.split('@')[0] for x in adata.obs.index.values]
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


