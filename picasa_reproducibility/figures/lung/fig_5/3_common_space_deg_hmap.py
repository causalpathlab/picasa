import sys
import scanpy as sc
import matplotlib.pylab as plt
import seaborn as sns
import anndata as ad
import os 
import glob 
import numpy as np
import pandas as pd

sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
# sample = sys.argv[1] 
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'+sample


############ read original data as adata list


ddir = wdir+'/data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

picasa_data = batch_map


############ read model results as adata 

df_main = pd.DataFrame()
for p_ad in picasa_data:
    df_main = pd.concat([df_main,picasa_data[p_ad].to_df()],axis=0)
    

wdir = wdir + '/fig_2/'
df = pd.read_csv(wdir+'/results/common_space_selected_cells.csv.gz',compression='gzip',index_col=0)
df.index = ['@'.join(x.split('@')[:2])for x in df.index.values]

df_main = df_main.loc[df.index.values]


adata = ad.AnnData(
    X=df_main.values,  
    obs=pd.DataFrame(index=df_main.index), 
    var=pd.DataFrame(index=df_main.columns)  
)

adata.obs['cluster'] = df['cluster'].values


import scanpy as sc

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
plt.savefig(wdir+'/results/deg_hmap_all.png')
plt.close()
