import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('data/figure2_attention_scores.csv.gz',index_col=0)
df = df.sample(n=50000)
print(df.shape)
##############################################

dfl = pd.DataFrame(df.index.values)
dfl['patient'] = [x.split('@')[0] for x in dfl[0]]
dfl['celltype'] = [x.split('@')[1].split('_')[0] for x in dfl[0]]
dfl['gene'] = [x.split('_')[1] for x in dfl[0]]
dfl.drop(0,axis=1,inplace=True)
dfl.reset_index(inplace=True)

##############################################
def get_top_genes_per_group(df,dfl,unique_celltypes,top_n):
    top_genes = {}
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = dfl[dfl['celltype'] == ct].index.values
        df_attn = df.iloc[ct_ylabel,:].copy()
        
        df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
        df_attn = df_attn.groupby('gene').mean()

        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        df_attn = df_attn.iloc[:top_n,:]
        tp1 = df_attn['gene'].unique()[:top_n]
        tp0 = df_attn['level_0'].unique()[:top_n]
        top_genes[ct]=np.concatenate([tp0,tp1])        
    return top_genes
        


unique_celltypes = dfl['celltype'].unique()
top_n = 2000
marker = get_top_genes_per_group(df,dfl,unique_celltypes,top_n)

seq_marker = []
for m in marker.keys(): 
    for x in marker[m]: seq_marker.append(x)

fig, axes = plt.subplots(5, 2, figsize=(10, 12))

for idx, ct in enumerate(unique_celltypes):
    
    row, col = idx // 2, idx % 2
    
    ct_ylabel = dfl[dfl['celltype'] == ct].index.values
    df_attn = df.iloc[ct_ylabel,:].copy()
    df_attn[df_attn > .001] = .001

    sel_genes = [x for x in seq_marker if x in df_attn.columns]
    df_attn = df_attn.loc[:,sel_genes]
    
    df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
    df_attn = df_attn[df_attn['gene'].isin(seq_marker)]
    df_attn = df_attn.groupby('gene').mean()
    # df_attn = df_attn.loc[sel_genes,sel_genes]
    
    df_attn.columns = [x.split('-')[0] for x in df_attn.columns]
    df_attn.index = [x.split('-')[0] for x in df_attn.index]
    sns.heatmap(df_attn, ax=axes[row, col],
                # yticklabels=df_attn.index,  
                # xticklabels=df_attn.columns,  
                cmap='viridis' 
                )
    axes[row, col].set_title(ct)
    
plt.tight_layout()
plt.savefig('results/figure2_attention_all_genes.png')
plt.close()
