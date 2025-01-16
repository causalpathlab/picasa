import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


sample ='pancreas'
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

p1 = 'indrop3'
p2 = 'smartseq2'


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
    top_genes = {}
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
        tp1 = df_attn['level_1'].unique()[:top_n]
        tp0 = df_attn['level_0'].unique()[:top_n]
        top_genes[ct]=np.concatenate([tp0,tp1])
        
    return top_genes


def plot_attention_group_wise(main_attn,main_y,marker):
    import matplotlib.pylab as plt
    import seaborn as sns
    from scipy.stats import zscore
    
    
            
    for idx, ct in enumerate(unique_celltypes):
        
        if ct not in marker:
            continue
        
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        # np.fill_diagonal(df_attn.values, 0)

        df_attn[df_attn > .01] = .01


        df_attn = df_attn.loc[:,marker[ct]]
        df_attn = df_attn.loc[marker[ct],:]
  
        # sns.clustermap(df_attn, cmap='viridis')
        # sns.map(df_attn, cmap='viridis')
        
        df_attn.columns = [x.split('-')[0] for x in df_attn.columns]
        df_attn.index = [x.split('-')[0] for x in df_attn.index]
        sns.heatmap(df_attn, 
                   yticklabels=df_attn.index,  
                   xticklabels=df_attn.columns,  
                   cmap='viridis' 
                   )
        plt.tight_layout()
        plt.savefig(wdir + '/results/picasa_common_attention_'+ct+'.png')
        plt.close()


marker = {
    'alpha' : ["GCG", "LOXL4", "PLCE1", "IRX2", "GC", "KLHL41", "CRYBA2", "TTR", "TM4SF4", "RGS4", "IRX2", "FEV", 
    "ARX", "PTGER3", "HMGB3", "RFX6", "MAFB", "SMARCA1", "PGR", "LDB2" ],
    'beta':["INS", "IAPP", "MAFA", "NPTX2", "DLK1", "ADCYAP1", "PFKFB2", "PDX1", "TGFBR3", "SYT13", "SMAD9", 
    "CDKN1C", "TFCP2L1", "SIX3", "SIX2", "MNX1", "BMP5", "PIR"],
    'delta':["SST", "HHEX", "PRG4", "ERBB4", "LEPR", "POU3F1", "RBP4", "ISL1", "BCHE", "PSIP1", "HHHEX", 
    "BHLHE41", "FRZB", "PDLIM4", "PCSK1", "EHF", "RGS2", "LCORL", "GABRG2", "ETV1"]
}


top_genes = get_top_genes_per_group(main_attn,main_y)

for m in ["alpha","beta","delta"]:
    pm = [x for x in marker[m] if x in adata_p1.var.index.values]
    marker[m] = np.concatenate([top_genes[m],pm])
    marker[m] = np.unique(marker[m])



plot_attention_group_wise(main_attn,main_y,marker=marker)






