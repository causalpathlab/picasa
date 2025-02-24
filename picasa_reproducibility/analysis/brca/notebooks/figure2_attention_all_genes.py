import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


df = pd.read_hdf('data/figure2_attention_scores.h5', key='df')

##############################################

dfl = pd.DataFrame(df.index.values)
dfl['patient'] = [x.split('@')[0] for x in dfl[0]]
dfl['celltype'] = [x.split('@')[1].split('_')[0] for x in dfl[0]]
dfl['gene'] = [x.split('_')[1] for x in dfl[0]]
dfl.drop(0,axis=1,inplace=True)
dfl.reset_index(inplace=True)

unique_celltypes = dfl['celltype'].unique()

##############################################
def get_top_genes_per_group(df,dfl,unique_celltypes):
    top_genes = {}
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = dfl[dfl['celltype'] == ct].index.values
        df_attn = df.iloc[ct_ylabel,:].copy()
        
        df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
        df_attn = df_attn.groupby('gene').mean()
        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        tp1 = df_attn['gene'].unique()
        top_genes[ct]=tp1
    return top_genes
        


marker = get_top_genes_per_group(df,dfl,unique_celltypes)
seq_marker = marker['Common0/Malignant']

fig, axes = plt.subplots(5, 2, figsize=(10, 12))

for idx, ct in enumerate(unique_celltypes):
    
    print(ct)
    
    row, col = idx // 2, idx % 2
    
    ct_ylabel = dfl[dfl['celltype'] == ct].index.values
    df_attn = df.iloc[ct_ylabel,:].copy()
    

    df_attn.index = [x.split('_')[1] for x in df_attn.index.values]
    df_attn = df_attn.loc[seq_marker,seq_marker]
    
    df_attn[df_attn > 0.0001] = 0.0001

    # df_attn = zscore(df_attn, axis=0)
    # th = 4
    # df_attn[df_attn > th] = th
    # df_attn[df_attn < -th] = -th
    sns.heatmap(df_attn, ax=axes[row, col],
                # yticklabels=df_attn.index,  
                # xticklabels=df_attn.columns,  
                cmap='viridis' 
                )
    axes[row, col].set_title(ct)
    
plt.tight_layout()
plt.savefig('results/figure2_attention_all_genes.png')
plt.close()
