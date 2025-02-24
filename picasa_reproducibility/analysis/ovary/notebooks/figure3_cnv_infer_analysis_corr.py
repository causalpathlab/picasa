import scanpy as sc
import infercnvpy as cnv
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
plt.savefig('results/figure3_cnv_corr.png')
plt.close()