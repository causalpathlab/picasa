import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('data/figure2_attention_scores.csv.gz',index_col=0)

df.isna().any().any()

##############################################

dfl = pd.DataFrame(df.index.values)
dfl['patient'] = [x.split('@')[0] for x in dfl[0]]
dfl['celltype'] = [x.split('@')[1].split('_')[0] for x in dfl[0]]
dfl['gene'] = [x.split('_')[1] for x in dfl[0]]
dfl.drop(0,axis=1,inplace=True)
dfl.reset_index(inplace=True)

marker = {
    'Common1/Malignant': ['EPCAM', 'WFDC2','SFTPA1', 'KRT6A', 'KRT5', 'NAPSA', 'EGFR', 'SOX2', 'MYC', 'TP63', 'DSG3', 'MUC16', 'PAX8', 'CLDN3','PAX8'],
    
    'Common10/Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'ACTA2', 'LUM', 'C1R', 'FAP','FGFR1','PDPN','MMP1'],
    
    'Common15/Endothelial': ['THBD','VWF', 'PECAM1', 'CLDN5', 'FLT1', 'KDR', 'CDH5', 'ANGPT2', 'ACKR1', 'GJA5', 'PROX1', 'PDPN', 'ESM1'],
    
    
    'Common0/T': ['CD2', 'CD3D', 'CD3E', 'CD3G','CD25','TRBC2', 'CD4','CD8A', 'TRAC', 'NKG7', 'GNLY', 'GZMA', 'GZMK', 'GZMB', 'GZMH','CD79A', 'FCER1G', 'PTPRC'],
    
    'Common7/Monocyte': ['CD14', 'CD68', 'LYZ', 'FCGR3A', 'FCGR1A', 'CD163', 'MRC1', 'FCN1','CD79A', 'FCER1G', 'PTPRC'],
    
    'Common8/B': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGKC', 'PAX5','CD79A', 'FCER1G', 'PTPRC'],
    
    'Common36/NK': ['NCAM1', 'KLRD1', 'KLRB1', 'KIR2DL4', 'NKG2A', 'PRF1','CD79A', 'FCER1G', 'PTPRC'],
    
    'Common17/DC': ['CD1C', 'CLEC10A', 'ITGAX', 'BATF3', 'HLA-DRA', 'CD86','CD79A', 'FCER1A', 'PTPRC'],
    
    'Common18/Mast': ['TPSAB1', 'TPSB2', 'CPA3', 'KIT', 'FCER1A', 'HDC','CD79A', 'FCER1G', 'PTPRC'],
    
    'Common19/Plasma': ['CD38', 'SDC1', 'MZB1', 'XBP1', 'IGHG1', 'IGHG3', 'PRDM1','CD79A', 'FCER1G', 'PTPRC']
}
 

unique_celltypes = dfl['celltype'].unique()

fig, axes = plt.subplots(5, 2, figsize=(10, 20))

for idx, ct in enumerate(unique_celltypes):
    
    row, col = idx // 2, idx % 2
    
    ct_ylabel = dfl[dfl['celltype'] == ct].index.values
    df_attn = df.iloc[ct_ylabel,:].copy()
    print(df_attn.shape)
    
    if ct == 'Epithelial' or ct == 'CD8T':
        df_attn[df_attn > .0001] = .0001
    else:
        df_attn[df_attn > .001] = .001

    sel_genes = [x for x in marker[ct] if x in df_attn.columns]
    df_attn = df_attn.loc[:,sel_genes]
    
    df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
    df_attn = df_attn[df_attn['gene'].isin(sel_genes)]
    df_attn = df_attn.groupby('gene').mean()
    
    df_attn = df_attn.loc[sel_genes,sel_genes]
    
    df_attn.columns = [x.split('-')[0] for x in df_attn.columns]
    df_attn.index = [x.split('-')[0] for x in df_attn.index]
    sns.heatmap(df_attn, ax=axes[row, col],
                yticklabels=df_attn.index,  
                xticklabels=df_attn.columns,  
                cmap='viridis' 
                )
    axes[row, col].set_title(ct)
    
plt.tight_layout()
plt.savefig('results/figure2_attention_celltype_markers.pdf')
plt.close()
