import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from plotnine import * 


adata = ad.read_h5ad('data/figure3_cancer_common_selected_cells.h5ad')
####################################


###### use normalized expression
# adata.X = np.expm1(adata.X)
# adata.X = np.log10(adata.X + 1)
# sc.pp.scale(adata)

# all_deg_clusters = []
# top_n = 100
# for clust in adata.obs['cluster'].unique():
# 	all_deg_clusters.append(adata[adata.obs['cluster']==clust].to_df().mean().sort_values(ascending=False).index.values[:top_n])

# all_deg_clusters = np.unique(np.array(all_deg_clusters).flatten())

# df_exp = adata.to_df()
# df_exp = df_exp[all_deg_clusters]

# n=250
# id_pairs = adata.obs.groupby('cluster').apply(lambda x: x.sample(n=min(n, len(x)))).index.values

# ids = [ x[1] for x in id_pairs]
# clust_ids = [ x[0] for x in id_pairs]
# df_exp = df_exp.loc[ids]
# df_exp.index = clust_ids
# df_exp = df_exp.T
# # df_exp[df_exp > 2] = 2
# sns.clustermap(df_exp,
# 	# yticklabels=df_deg.index,  
#     # xticklabels=df_deg.columns,
#     col_cluster=False,  
# cmap=sns.color_palette("coolwarm", as_cmap=True))
# plt.title(" score")
# plt.xticks(rotation=90)
# plt.savefig('results/figure3_cancer_common_exp_hmap.png')
# plt.close()

###### use normalized expression
sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon')
deg_results = adata.uns['rank_genes_groups'] 

all_deg_dfs = []
for group in adata.obs['cluster'].unique(): 
    deg_df = sc.get.rank_genes_groups_df(adata, group=group)
    deg_df.sort_values('logfoldchanges',ascending=False,inplace=True)
    deg_df['cluster'] = group  
    all_deg_dfs.append(deg_df)
all_deg_df = pd.concat(all_deg_dfs, ignore_index=True)

deg_filtered = all_deg_df[all_deg_df['pvals_adj'] < 0.01]
deg_filtered = deg_filtered[['names','logfoldchanges','cluster']]

df_deg = deg_filtered.pivot(index='names', columns='cluster', values='logfoldchanges')
df_deg = df_deg.replace([np.inf, -np.inf], np.nan) 
df_deg = df_deg.fillna(0) 
df_deg[df_deg > 2] = 2
df_deg[df_deg < -2] = -2

# df_deg = df_deg.T
sns.clustermap(df_deg,
	# yticklabels=df_deg.index,  
    # xticklabels=df_deg.columns,  
cmap=sns.color_palette("coolwarm", as_cmap=True))
plt.title(" score")
plt.xticks(rotation=90)
plt.savefig('results/figure3_cancer_common_deg_hmap.png')
plt.close()
