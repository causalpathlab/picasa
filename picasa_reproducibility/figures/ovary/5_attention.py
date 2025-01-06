import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


sample ='ovary'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

############ read original data as adata list
import os 
import glob 

ddir = pp+sample+'/data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

picasa_data = batch_map


############ read model results as adata 
wdir = pp+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')




from picasa import model,dutil
import torch

nn_params = picasa_adata.uns['nn_params']
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/results/picasa_common.model', map_location=torch.device(nn_params['device'])))


# p1 = picasa_adata.uns['adata_keys'][0]
# p2 = picasa_adata.uns['adata_keys'][1]

p1 = 'EOC3'
p2 = 'EOC153'

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


unique_celltypes = adata_p1.obs['celltype'].unique()
num_celltypes = len(unique_celltypes)

def get_top_genes_per_group(main_attn,main_y):
    top_genes = []
    top_n = 3
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        np.fill_diagonal(df_attn.values, 0)
        ## note that prev column is now first column 
        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        top_genes.append(df_attn['level_1'].unique()[:top_n])
        print(ct,df_attn['level_1'].unique()[:top_n])
        
    top_genes = np.unique(np.array(top_genes).flatten())
    return top_genes


def plot_attention_group_wise(main_attn,main_y,mode='top_genes',marker=None):
    import matplotlib.pylab as plt
    import seaborn as sns
    from scipy.stats import zscore
    
    top_genes = []
    
    if mode == 'top_genes':
    
        top_genes = get_top_genes_per_group(main_attn,main_y)
    
    elif mode == 'marker':
        
        top_genes = marker
        
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        np.fill_diagonal(df_attn.values, 0)

        df_attn = df_attn.apply(zscore)
        df_attn[df_attn > 10] = 10
        # df_attn[df_attn < -0.1] = -0.1
        df_attn = df_attn.loc[:,top_genes]
        df_attn = df_attn.loc[top_genes,:]
  
        sns.heatmap(df_attn, cmap='viridis')
        # sns.map(df_attn, cmap='viridis')
        plt.tight_layout()
        plt.savefig(wdir + '/results/picasa_common_attention_'+ct+'.png')
        plt.close()


marker = np.array(['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB'])

plot_attention_group_wise(main_attn,main_y,mode='top_genes')




