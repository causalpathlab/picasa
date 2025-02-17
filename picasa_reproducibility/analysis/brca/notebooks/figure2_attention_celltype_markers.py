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
    'CD4Tconv': ['CD3D', 'CD3E', 'CD4', 'IL7R', 'CCR7','FOXP3','CD25','CXCL13', 'IL21','PDCD1'],
    'CD8Tex': ['CD8A', 'CD8B', 'LAG3', 'PDCD1', 'TIGIT', 'EOMES', 'TBX21','IFNG','TNF','ZFP36'],
    'Tprolif': ['MKI67', 'TOP2A', 'CD3D', 'CD3E', 'CD8A', 'CD8B'],
    'B': ['IGKC','CD27','IGHD','IGLC2', 'CD19', 'MS4A1', 'CD79A', 'PAX5'],
    'Mono/Macro': ['CD68', 'CD14', 'FCGR3A', 'CSF1R', 'MRC1','IL1B','S100A9','CD16'],
    'DC': ['CD1C', 'CLEC9A', 'BATF3', 'ITGAX', 'HLA-DQB1', 'IRF7', 'LAMP3', 'XCR1', 'CCR7', 'HLA-DRA', 'FCER1A', 'IL3RA', 'CD86', 'CD80', 'FLT3'],
    
    'Malignant': ['EPCAM', 'KRT8', 'KRT6B', 'KRT14', 'MKI67', 'TP53', 'SOX9',
                  'CXCL13','S100A1','TRH','EMC3','PIP','HPD','HULC','MARCO','TFF1'
                  ],
    'Plasma': ['SDC1', 'CD38', 'IGHG1', 'MZB1', 'XBP1'],
    'SMC': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'MYL9'],
    'Endothelial': ['ACKR1', 'SELE', 'SELP','VWF', 'PECAM1', 'CDH5', 'FLT1', 'KDR','ICAM1' ,'VCAM1',
                    'DLL4','RGS5','CXCL12','VEGFC'
                    ],
    'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN', 'PDGFRB','PDGFRA', 'THY1'],
}

unique_celltypes = dfl['celltype'].unique()

fig, axes = plt.subplots(6, 2, figsize=(10, 20))

for idx, ct in enumerate(unique_celltypes):
    
    row, col = idx // 2, idx % 2
    
    ct_ylabel = dfl[dfl['celltype'] == ct].index.values
    df_attn = df.iloc[ct_ylabel,:].copy()
    print(df_attn.shape)
    

    sel_genes = [x for x in marker[ct] if x in df_attn.columns]
    df_attn = df_attn.loc[:,sel_genes]
    
    df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
    df_attn = df_attn[df_attn['gene'].isin(sel_genes)]
    df_attn = df_attn.groupby('gene').mean()
    
    df_attn = df_attn.loc[sel_genes,sel_genes]
    
    sns.heatmap(df_attn, ax=axes[row, col],
                yticklabels=df_attn.index,  
                xticklabels=df_attn.columns,  
                cmap='viridis'
                )
    axes[row, col].set_title(ct)
    
plt.tight_layout()
plt.savefig('results/figure2_attention_celltype_markers.pdf')
plt.close()
