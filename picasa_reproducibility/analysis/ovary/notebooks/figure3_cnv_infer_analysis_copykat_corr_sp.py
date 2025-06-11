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

sample ='ovary'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample

### prep raw data
adata = ad.read_h5ad(wdir+'/data/all_ovary.h5ad')
patients = adata.obs['batch'].unique()
celltypes = adata.obs['celltype'].unique()


pattern = '*_copykat_CNA_results.txt'

file_paths = glob.glob(pattern)

df_fp = pd.DataFrame(file_paths,columns=['filepath'])
df_fp['fpc'] = [x.replace('__copykat_CNA_results.txt','').replace('raw_','').replace('recons_','') for x in df_fp['filepath']]
df_fp['patient'] = [x.split('_')[0] for x in df_fp['fpc']]
df_fp['treatment'] = [x.split('_')[1] for x in df_fp['fpc']]
df_fp['celltype'] = [x.split('_')[2] for x in df_fp['fpc']]

df_fp_count = df_fp.groupby(['patient','treatment','celltype']).count().reset_index()

df_fp_count = df_fp_count[df_fp_count['fpc']==2]
ptct_pairs = [(x,y,z) for x,y,z in zip(
    df_fp_count['patient'],
    df_fp_count['treatment'],
    df_fp_count['celltype']
)]


# res = []
# for ptct_pair in ptct_pairs:
#     pc = ptct_pair[0]        
#     tc = ptct_pair[1]        
#     ctc = ptct_pair[2]
    
#     df_raw = pd.read_csv('raw_'+pc+'_'+tc+'_'+ctc+'__copykat_CNA_results.txt',sep='\t')    
#     cols = [str(x)+'_'+str(y) for x,y in zip(df_raw['chrom'],df_raw['chrompos'])]
#     df_raw = df_raw.iloc[:,3:]
#     df_raw = df_raw.T
#     df_raw.columns = cols

#     df_recons = pd.read_csv('recons_'+pc+'_'+tc+'_'+ctc+'__copykat_CNA_results.txt',sep='\t')
#     cols = [str(x)+'_'+str(y) for x,y in zip(df_recons['chrom'],df_recons['chrompos'])]
#     df_recons = df_recons.iloc[:,3:]
#     df_recons = df_recons.T
#     df_recons.columns = cols

#     org_vals = df_raw.mean().values
#     rec_vals = df_recons.mean().values
#     corval = round(spearmanr(org_vals,rec_vals).statistic,5)
            
#     res.append([pc,tc,ctc,corval])
            

# pd.DataFrame(res,columns=['patient','treatment','celltype','cnv_corr']).to_csv('figure3_cnv_celltype_corr_sp_copykat.csv.gz',compression='gzip')


# ### plot patient wise
# cutoff = 3
# unique_patients = df_fp['patient'].unique()
# cn = 3
# rn = int(np.ceil(len(unique_patients) / cn))  

# fig, axes = plt.subplots(rn, cn, figsize=(20, 30))

# for idx, p in enumerate(unique_patients):
    
#     row, col = divmod(idx, cn)
#     print(row,col)

#     pattern = 'raw_'+p+'*__copykat_CNA_results.txt'
#     file_paths = glob.glob(pattern)
#     df_patient_raw = pd.DataFrame()
#     for f in file_paths:
#         df_raw_c = pd.read_csv(f,sep='\t')    
#         cols = [str(x)+'_'+str(y) for x,y in zip(df_raw_c['chrom'],df_raw_c['chrompos'])]
#         df_raw_c = df_raw_c.iloc[:,3:]
#         df_raw_c = df_raw_c.T
#         df_raw_c.columns = cols
        
#         df_patient_raw = pd.concat([df_patient_raw,df_raw_c])

#     pattern = 'recons_'+p+'*__copykat_CNA_results.txt'
#     file_paths = glob.glob(pattern)
#     df_patient_recons = pd.DataFrame()
#     for f in file_paths:
#         df_recons_c = pd.read_csv(f,sep='\t')    
#         cols = [str(x)+'_'+str(y) for x,y in zip(df_recons_c['chrom'],df_recons_c['chrompos'])]
#         df_recons_c = df_recons_c.iloc[:,3:]
#         df_recons_c = df_recons_c.T
#         df_recons_c.columns = cols
        
#         df_patient_recons = pd.concat([df_patient_recons,df_recons_c])



#     org_vals = df_patient_raw.mean().values
#     rec_vals = df_patient_recons.mean().values
#     corval = round(spearmanr(org_vals,rec_vals).correlation,5)
    
#     df_plot = pd.DataFrame({
#         'Original': org_vals,
#         'Reconstructed': rec_vals
#     })

#     print(corval)
    
#     df_z = zscore(df_plot, axis=0)
#     nonoutlier_idxs = df_z[   
#             (df_z['Original']<cutoff) &
#             (df_z['Reconstructed']<cutoff) &
#             (df_z['Original']>-cutoff) &
#             (df_z['Reconstructed']>-cutoff) 
#         ].index.values

#     df_plot = df_plot.iloc[nonoutlier_idxs]


#     ax = sns.kdeplot(
#         data=df_plot, 
#         x="Original", 
#         y="Reconstructed", 
#         cmap="coolwarm",  
#         levels=10,
#         ax = axes[row,col],
#         fill=True  
#     )
    
#     axes[row, col].set_title(p)
#     axes[row, col].text(
#         0.5, 0.9, corval, 
#         transform=axes[row, col].transAxes, 
#         fontsize=12, 
#         color='black', 
#         weight='bold', 
#         ha='left', va='top'
#     )
#     axes[row, col].set_xticks([])
#     axes[row, col].set_yticks([])
#     axes[row, col].set_xlabel('')
#     axes[row, col].set_ylabel('')

# plt.tight_layout()
# plt.savefig('figure3_cnv_analysis_scatter_patient_all_sp_copykat.pdf')
# plt.close()


for p in  df_fp_count['patient'].unique():

    for treatment in  df_fp_count['treatment'].unique():
        
        unique_celltypes = df_fp_count.loc[ (df_fp_count['patient']==p) & (df_fp_count['treatment']==treatment)]['celltype']
        cutoff = 3
            
        cn = len(unique_celltypes)
        rn = 1  
        if cn == 1 : continue
        
        fig, axes = plt.subplots(rn, cn, figsize=(20, 10))
        
        for idx, ct in enumerate(unique_celltypes):
            
            col = idx
            row = 0
            
            print(row,col)

            pattern = 'raw_'+p+'_'+treatment+ '_'+ct+'__copykat_CNA_results.txt'
            file_paths = glob.glob(pattern)
            df_patient_raw = pd.DataFrame()
            for f in file_paths:
                df_raw_c = pd.read_csv(f,sep='\t')    
                cols = [str(x)+'_'+str(y) for x,y in zip(df_raw_c['chrom'],df_raw_c['chrompos'])]
                df_raw_c = df_raw_c.iloc[:,3:]
                df_raw_c = df_raw_c.T
                df_raw_c.columns = cols
                
                df_patient_raw = pd.concat([df_patient_raw,df_raw_c])

            pattern = 'recons_'+p+'_'+treatment+ '_'+ct+'__copykat_CNA_results.txt'
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
            corval = round(spearmanr(org_vals,rec_vals).correlation,3)
            
            df_plot = pd.DataFrame({
                'Original': org_vals,
                'Reconstructed': rec_vals
            })

            print(p,ct,corval)
            
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
                ax = axes[col],
                fill=True  
            )
            
            axes[col].set_title(ct)
            axes[col].text(
                0.5, 0.9, corval, 
                transform=axes[col].transAxes, 
                fontsize=12, 
                color='black', 
                weight='bold', 
                ha='left', va='top'
            )
            
            axes[col].set_xticks([])
            axes[col].set_yticks([])
            axes[col].set_xlabel('')
            axes[col].set_ylabel('')

        
        plt.tight_layout()
        plt.savefig('figure3_cnv_analysis_scatter_patient_'+p+'_'+treatment+ '_'+'_copykat.pdf')
        plt.close()



## print cell type table

import pandas as pd
df = pd.read_csv('figure3_cnv_celltype_corr_sp_copykat.csv.gz')
dfg = df.groupby(['patient','treatment','celltype'])['cnv_corr'].mean()
dfg = dfg.reset_index()
dfg = dfg.sort_values('cnv_corr',ascending=False)
dfg.to_csv('figure3_cnv_celltype_corr_sp_copykat_celltype.csv.gz',compression='gzip')

