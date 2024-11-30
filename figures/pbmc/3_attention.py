import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa'
sample = 'pbmc'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_pbmc1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_pbmc2.h5ad')
picasa_data = {'pbmc1':batch1,'pbmc2':batch2}


############ read model results as adata 
wdir = pp+'/figures/'+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')


############ add metadata
dfl= pd.read_csv(ddir+sample+'_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]
dfl = dfl[['index','cell','celltype']]
picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')



from picasa import model,dutil
import torch

nn_params = picasa_adata.uns['nn_params']
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/results/picasa_common.model', map_location=torch.device(nn_params['device'])))


p1 = 'pbmc1'
p2 = 'pbmc2'
adata_p1 = picasa_data[p1]
adata_p2 = picasa_data[p2]
df_nbr = picasa_adata.uns['nbr_map']
df_nbr = df_nbr[df_nbr['batch_pair']==p1+'_'+p2]
nbr_map = {x:(y,z) for x,y,z in zip(df_nbr['key'],df_nbr['neighbor'],df_nbr['score'])}

data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map,nn_params['device'],batch_size=100)

main_attn,main_y = model.eval_attention_common(picasa_common_model,data_loader)


##############################################


import numpy as np 


unique_celltypes = adata_p1.obs['celltype'].unique()
num_celltypes = len(unique_celltypes)

def get_top_genes_per_group(main_attn,main_y):
    top_genes = []
    top_n = 10
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        
        ## note that prev column is now first column 
        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        top_genes.append(df_attn['level_1'].unique()[:top_n])
    top_genes = np.unique(np.array(top_genes).flatten())
    return top_genes

def plot_attention_group_wise(main_attn,main_y,mode='top_genes',marker=None):
    import matplotlib.pylab as plt
    import seaborn as sns
    from scipy.stats import zscore
    
    top_genes = []
    
    if mode == 'top_genes':
    
        top_genes = get_top_genes_per_group(main_attn,main_y)
    
    if mode == 'marker':
        
        top_genes = marker
        
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        
        df_attn = df_attn.apply(zscore)
        df_attn[df_attn > 5] = 5
        df_attn[df_attn < -5] = -5

        df_attn = df_attn.loc[:,top_genes]
        df_attn = df_attn.loc[top_genes,:]
  
        sns.clustermap(df_attn, cmap='viridis')
        plt.tight_layout()
        plt.savefig(wdir + '/results/picasa_common_attention_'+ct+'.png')
        plt.close()

plot_attention_group_wise(main_attn,main_y)

marker = np.array(['IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A',
	'GNLY', 'NKG7', 'CST3', 'CD3E', 'FCER1A', 'CD74', 'LST1', 'CCL5',
	'HLA-DPA1', 'LDHB', 'CD79A', 'FCER1G', 'GZMB', 'S100A9',
	'HLA-DPB1', 'HLA-DRA', 'AIF1', 'CST7', 'S100A8', 'CD79B', 'COTL1',
	'CTSW', 'B2M', 'TYROBP', 'HLA-DRB1', 'PRF1', 'GZMA', 'FTL', 'NRGN'])

plot_attention_group_wise(main_attn,main_y,mode='marker',marker=marker)