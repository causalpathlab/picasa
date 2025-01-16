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


all_deg_df['pvals_adj'] = all_deg_df['pvals_adj'] + 1e-8
all_deg_df['log_pvals_adj'] = -np.log10(all_deg_df['pvals_adj'])

logfc_threshold = 2.0
pval_threshold = 0.01
all_deg_df['significance'] = 'Not Significant'
all_deg_df.loc[
    (all_deg_df['logfoldchanges'] > logfc_threshold) & (all_deg_df['pvals_adj'] < pval_threshold),
    'significance'
] = 'Upregulated'


all_deg_df.loc[
    (all_deg_df['logfoldchanges'] < -logfc_threshold) & (all_deg_df['pvals_adj'] < pval_threshold),
    'significance'
] = 'Downregulated'


plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=all_deg_df,
    x='logfoldchanges',
    y='log_pvals_adj',
    hue='cluster',  
    palette='tab10', 
    alpha=0.7
)

# for i, row in all_deg_df.iterrows():
#     plt.text(
#         x=row['logfoldchanges'], 
#         y=row['log_pvals_adj'], 
#         s=row['names'],        
#         fontsize=8,            
#         alpha=0.8           
#     )
    
    
plt.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', linewidth=1)
plt.axvline(x=logfc_threshold, color='black', linestyle='--', linewidth=1)
plt.axvline(x=-logfc_threshold, color='black', linestyle='--', linewidth=1)

plt.title('Volcano Plot of DEGs by Cluster', fontsize=16)
plt.xlabel('Log2 Fold Change', fontsize=14)
plt.ylabel('-Log10 Adjusted P-value', fontsize=14)
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(wdir + '/results/volcano_plot_by_cluster.png')
plt.show()
