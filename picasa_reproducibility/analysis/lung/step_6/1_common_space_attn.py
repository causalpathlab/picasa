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
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')


df = pd.read_csv(wdir+'/fig_2/results/common_space_selected_cells.csv.gz',compression='gzip',index_col=0)
df.index = ['@'.join(x.split('@')[:2])for x in df.index.values]




from picasa import model,dutil
import torch

nn_params = picasa_adata.uns['nn_params']
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/results/picasa_common.model', map_location=torch.device(nn_params['device'])))


p1 = 'P6'
p2 = 'P3'

adata_p1 = picasa_data[p1]
adata_p2 = picasa_data[p2]
df_nbr = picasa_adata.uns['nbr_map']
df_nbr = df_nbr[df_nbr['batch_pair']==p1+'_'+p2]
nbr_map = {x:(y,z) for x,y,z in zip(df_nbr['key'],df_nbr['neighbor'],df_nbr['score'])}

data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map,'cpu',batch_size=10)
eval_total_size=1000
main_attn,main_y = model.eval_attention_common(picasa_common_model,data_loader,eval_total_size)


##############################################


import numpy as np 

wdir = wdir+'/fig_3/'

unique_celltypes = df['cluster'].unique()
num_celltypes = len(unique_celltypes)


df = df[df['batch']==p1]

def get_top_genes_per_group(main_attn,main_y):
    top_genes = []
    top_n = 3
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = df[df['cluster']==ct].index.values
        print(ct,len(ct_ylabel))
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        np.fill_diagonal(df_attn.values, 0)
        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        tp1 = df_attn['level_1'].unique()[:top_n]
        tp0 = df_attn['level_0'].unique()[:top_n]
        top_genes.append(np.concatenate([tp0,tp1]))
        
    return top_genes

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_attention_group_wise(main_attn, main_y, unique_celltypes, df, adata_p1, top_genes, wdir):
    n_cells = len(unique_celltypes)
    n_cols = 3
    n_rows = (n_cells // n_cols) + (1 if n_cells % n_cols != 0 else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = df[df['cluster'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))
        
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                               index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        np.fill_diagonal(df_attn.values, 0)

        df_attn[df_attn > 0.001] = 0.001

        df_attn = df_attn.loc[:, top_genes]
        df_attn = df_attn.loc[top_genes, :]
        
        df_attn.columns = [x.split('-')[0] for x in df_attn.columns]
        df_attn.index = [x.split('-')[0] for x in df_attn.index]
        
        sns.heatmap(df_attn, ax=axes[idx], yticklabels=df_attn.index, xticklabels=df_attn.columns, cmap='viridis')
        axes[idx].set_title(ct)
    
    plt.tight_layout()
    plt.savefig(wdir + '/results/picasa_common_attention.png')
    plt.close()


top_genes = get_top_genes_per_group(main_attn,main_y)

top_genes = np.unique(np.array(top_genes))

plot_attention_group_wise(main_attn, main_y, unique_celltypes, df, adata_p1, top_genes, wdir)


