import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('data/figure2_attention_scores.csv.gz',index_col=0)

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
top_n = 10
marker = get_top_genes_per_group(df,dfl,unique_celltypes,top_n)

dftop = pd.DataFrame.from_dict(marker, orient='index')

dftop['Top genes'] = dftop.astype(str).agg(lambda x: ','.join(x), axis=1)

dftop = dftop[['Top genes']]

dftop.to_csv('results/figure2_attention_top_genes.csv')
