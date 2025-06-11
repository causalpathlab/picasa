import scanpy as sc
import anndata as ad
import pandas as pd 
import numpy as np


sample ='ovary'
pp = '/data/sishir/projects/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample

### prep raw data
adata = ad.read_h5ad(wdir+'/data/all_ovary.h5ad')
adata.X = np.expm1(adata.X)
df = adata.to_df()

df_obs = adata.obs

patients = df_obs['batch'].unique()
celltypes = df_obs['celltype'].unique()
treatments = df_obs['treatment_phase'].unique()

for p in patients:
    for t in treatments:
        for ct in celltypes:
            idxs = df_obs.loc[ (df_obs['batch']==p ) & (df_obs['treatment_phase']==t) & (df_obs['celltype']==ct) ].index.values
            df_c = df.loc[idxs,:]
            print(df_c.shape)
            if df_c.shape[0]>249:
                print('raw_data_'+p+'_'+t+'_'+ct)
                df_c.to_parquet('raw_data_'+p+'_'+t+'_'+ct+'.parquet')

### prep recons data
adata_unq = ad.read_h5ad(wdir+'/notebooks/data/figure3_unique_recons.h5ad')
# adata_unq.X = np.expm1(adata_unq.X)
df_unq = adata_unq.to_df()

for p in patients:
    for t in treatments:
        for ct in celltypes:
            idxs = df_obs.loc[ (df_obs['batch']==p ) & (df_obs['treatment_phase']==t) & (df_obs['celltype']==ct) ].index.values
            df_unq = df.loc[idxs,:]
            print(df_unq.shape)
            if df_unq.shape[0]>249:
                print('recons_data_'+p+'_'+t+'_'+ct)
                df_unq.to_parquet('recons_data_'+p+'_'+t+'_'+ct+'.parquet')



