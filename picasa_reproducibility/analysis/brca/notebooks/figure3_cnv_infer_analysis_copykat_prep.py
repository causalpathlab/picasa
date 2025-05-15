import scanpy as sc
import anndata as ad
import pandas as pd 
import numpy as np


sample ='brca'
pp = '/data/sishir/projects/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample

### prep raw data
adata = ad.read_h5ad(wdir+'/data/all_brca.h5ad')
adata.X = np.expm1(adata.X)
df = adata.to_df()

df_obs = adata.obs

patients = df_obs['batch'].unique()
df_obs['celltype'] = [x.replace('/','') for x in df_obs['celltype']]
celltypes = df_obs['celltype'].unique()

for p in patients:
        for ct in celltypes:
            idxs = df_obs.loc[ (df_obs['batch']==p ) & (df_obs['celltype']==ct) ].index.values
            df_c = df.loc[idxs,:]
            print(df_c.shape)
            if df_c.shape[0]>249:
                print('raw_data_'+p+'_'+ct)
                df_c.to_parquet('raw_data_'+p+'_'+ct+'.parquet')

### prep recons data
adata_unq = ad.read_h5ad(wdir+'/notebooks/data/figure3_unique_recons.h5ad')
df_unq = adata_unq.to_df()

for p in patients:
        for ct in celltypes:
            idxs = df_obs.loc[ (df_obs['batch']==p ) &  (df_obs['celltype']==ct) ].index.values
            df_unq = df.loc[idxs,:]
            print(df_unq.shape)
            if df_unq.shape[0]>249:
                print('recons_data_'+p+'_'+ct)
                df_unq.to_parquet('recons_data_'+p+'_'+ct+'.parquet')

