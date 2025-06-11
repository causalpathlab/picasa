import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from plotnine import * 
import sys
from scipy.stats import spearmanr
from scipy.stats import zscore
import os 
import glob

sample ='brca'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample

### prep raw data
adata = ad.read_h5ad(wdir+'/data/all_brca.h5ad')
patients = adata.obs['batch'].unique()
celltypes = adata.obs['celltype'].unique()


pattern = '*_copykat_CNA_results.txt'

file_paths = glob.glob(pattern)

df_fp = pd.DataFrame(file_paths,columns=['filepath'])
df_fp['fpc'] = [x.replace('__copykat_CNA_results.txt','').replace('raw_','').replace('recons_','') for x in df_fp['filepath']]
df_fp['patient'] = [x.split('_')[0] for x in df_fp['fpc']]
df_fp['celltype'] = [x.split('_')[1] for x in df_fp['fpc']]

df_fp_count = df_fp.groupby(['patient','celltype']).count().reset_index()

df_fp_count = df_fp_count[df_fp_count['fpc']==2]

ptct_pairs = [(x,y) for x,y in zip(
    df_fp_count['patient'],
    df_fp_count['celltype']
)]


# res = []
# for ptct_pair in ptct_pairs:
#     pc = ptct_pair[0]        
#     ctc = ptct_pair[1]
    
#     df_raw = pd.read_csv('raw_'+pc+'_'+ctc+'__copykat_CNA_results.txt',sep='\t')    
#     cols = [str(x)+'_'+str(y) for x,y in zip(df_raw['chrom'],df_raw['chrompos'])]
#     df_raw = df_raw.iloc[:,3:]
#     df_raw = df_raw.T
#     df_raw.columns = cols

#     df_recons = pd.read_csv('recons_'+pc+'_'+ctc+'__copykat_CNA_results.txt',sep='\t')
#     cols = [str(x)+'_'+str(y) for x,y in zip(df_recons['chrom'],df_recons['chrompos'])]
#     df_recons = df_recons.iloc[:,3:]
#     df_recons = df_recons.T
#     df_recons.columns = cols

#     org_vals = df_raw.mean().values
#     rec_vals = df_recons.mean().values
#     corval = round(spearmanr(org_vals,rec_vals).correlation,5)
            
#     res.append([pc,ctc,corval])
            

# df_res = pd.DataFrame(res,columns=['patient','celltype','cnv_corr'])
# subtype_map = {x:y for x,y in zip(adata.obs.batch,adata.obs.subtype)}
# df_res['subtype'] = [subtype_map[x] for x in df_res['patient']]
# df_res.to_csv('figure3_cnv_celltype_corr_sp_copykat.csv.gz',compression='gzip')


### plot patient wise
cutoff = 3
unique_patients = df_fp['patient'].unique()
cn = 5
rn = int(np.ceil(len(unique_patients) / cn))  

fig, axes = plt.subplots(rn, cn, figsize=(20, 30))

for idx, p in enumerate(unique_patients):
    
    row, col = divmod(idx, cn)
    print(row,col)

    pattern = 'raw_'+p+'*__copykat_CNA_results.txt'
    file_paths = glob.glob(pattern)
    df_patient_raw = pd.DataFrame()
    for f in file_paths:
        df_raw_c = pd.read_csv(f,sep='\t')    
        cols = [str(x)+'_'+str(y) for x,y in zip(df_raw_c['chrom'],df_raw_c['chrompos'])]
        df_raw_c = df_raw_c.iloc[:,3:]
        df_raw_c = df_raw_c.T
        df_raw_c.columns = cols
        
        df_patient_raw = pd.concat([df_patient_raw,df_raw_c])

    pattern = 'recons_'+p+'*__copykat_CNA_results.txt'
    file_paths = glob.glob(pattern)
    df_patient_recons = pd.DataFrame()
    for f in file_paths:
        df_recons_c = pd.read_csv(f,sep='\t')    
        cols = [str(x)+'_'+str(y) for x,y in zip(df_recons_c['chrom'],df_recons_c['chrompos'])]
        df_recons_c = df_recons_c.iloc[:,3:]
        df_recons_c = df_recons_c.T
        df_recons_c.columns = cols
        
        df_patient_recons = pd.concat([df_patient_recons,df_recons_c])



    org_vals = df_patient_raw.mean().values
    rec_vals = df_patient_recons.mean().values
    corval = round(spearmanr(org_vals,rec_vals).correlation,5)
    
    df_plot = pd.DataFrame({
        'Original': org_vals,
        'Reconstructed': rec_vals
    })

    print(corval)
    
    df_z = zscore(df_plot, axis=0)
    nonoutlier_idxs = df_z[   
            (df_z['Original']<cutoff) &
            (df_z['Reconstructed']<cutoff) &
            (df_z['Original']>-cutoff) &
            (df_z['Reconstructed']>-cutoff) 
        ].index.values

    df_plot = df_plot.iloc[nonoutlier_idxs]


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
    
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
    axes[row, col].set_xlabel('')
    axes[row, col].set_ylabel('')
    
plt.tight_layout()
plt.savefig('figure3_cnv_analysis_scatter_patient_all_sp_copykat.pdf')
plt.close()
