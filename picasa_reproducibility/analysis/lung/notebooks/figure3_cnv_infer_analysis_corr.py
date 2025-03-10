import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from plotnine import * 
import sys


df_org = pd.read_parquet(f'results/figure3_cnv_orig_profile.parquet', engine='pyarrow')
df_rec = pd.read_parquet(f'results/figure3_cnv_recons_profile.parquet', engine='pyarrow')


ci = np.intersect1d(df_org.index.values,df_rec.index.values)
df_org = df_org.loc[ci]
df_rec = df_rec.loc[ci]

correlation = df_org.T.corrwith(df_rec.T)
print(correlation.mean(),correlation.median())
sns.displot(correlation)
plt.savefig('results/figure3_cnv_analysis_corr.pdf')
plt.close()



### plot patient wise
picasa_adata = ad.read_h5ad('../model_results/picasa.h5ad')

df_obs = picasa_adata.obs.copy()
df_obs.index = ['@'.join(x.split('@')[:2]) for x in df_obs.index]
df_obs = df_obs.loc[df_org.index.values]

unique_patients = df_obs['batch'].unique()
cn = 5
rn = int(np.ceil(len(unique_patients) / cn))  

fig, axes = plt.subplots(rn, cn, figsize=(20, 30))

for idx, p in enumerate(unique_patients):
    
    row, col = divmod(idx, cn)
    print(row,col)

    cells = df_obs.loc[df_obs['batch']==p].index.values

    org_vals = df_org.loc[cells,:].mean().values
    rec_vals = df_rec.loc[cells,:].mean().values
    
    corval = str(np.corrcoef(org_vals,rec_vals)[0,1])[:5]
        
    df_plot = pd.DataFrame({
        'Original': org_vals,
        'Reconstructed': rec_vals
    })

    ax = sns.kdeplot(
        data=df_plot, 
        x="Original", 
        y="Reconstructed", 
        cmap="coolwarm",  
        levels=10,
        ax = axes[row,col],
        fill=True  
    )
    
    axes[row, col].set_title(p)
    axes[row, col].text(
        0.5, 0.9, corval, 
        transform=axes[row, col].transAxes, 
        fontsize=12, 
        color='black', 
        weight='bold', 
        ha='left', va='top'
    )
    
plt.tight_layout()
plt.savefig('results/figure3_cnv_analysis_scatter_patient_all.pdf')
plt.close()
