import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import torch
import numpy as np


sample ='lung'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'

############ read original data as adata list

ddir = pp+sample+'/model_data/'
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
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_lung.h5ad')

from scvi.distributions import ZeroInflatedNegativeBinomial
import scanpy as sc
import matplotlib.pyplot as plt
from picasa import model
import torch 

def get_zinb_reconstruction(x, px_s, px_r, px_d):
    zinb_dist = ZeroInflatedNegativeBinomial(mu=px_s, theta=px_r, zi_logits=px_d)
    reconstructed_x = zinb_dist.sample()
    return reconstructed_x


num_batches = len(picasa_adata.obs['batch'].unique())
input_dim = adata.shape[1]
nn_params = picasa_adata.uns['nn_params']
enc_layers = [128,25]
unique_latent_dim = nn_params['latent_dim']
common_latent_dim = nn_params['latent_dim']
dec_layers = [128,128]
nn_params['device'] = 'cpu'


picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(nn_params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+'/model_results/picasa_unique.model', map_location=torch.device(nn_params['device'])))

picasa_unique_model.eval()


for p1 in adata.obs['batch'].unique():
    
	current_adata = adata[adata.obs['batch']==p1].copy()
 
	df = current_adata.to_df()

	x_c1 = torch.tensor(df.values).float()
	x_z = torch.tensor(pd.DataFrame(np.zeros((df.shape[0],25))).values).float()
	y = current_adata.obs_names.values

	z,ylabel = model.predict_batch_unique(picasa_unique_model,x_c1,y,x_z)
	z_u = z[0]
	px_scale = z[1]
	px_rate = z[2]
	px_dropout = z[3]
	batch_pred = z[4]



	x_recons = get_zinb_reconstruction(x_c1,px_scale,px_rate,px_dropout)
	df_recons = pd.DataFrame(x_recons.detach().numpy(),index=df.index.values,columns = df.columns)
	df_recons.to_csv('data/figure5_cnv_x_recons_'+p1+'.csv.gz',compression='gzip')
	df.to_csv('data/figure5_cnv_x_orig_'+p1+'.csv.gz',compression='gzip')


	current_adata.obsm['unique'] = z_u.detach().numpy()
	sc.pp.neighbors(current_adata, use_rep="unique")
	sc.tl.umap(current_adata)
	sc.pl.umap(current_adata,color=['batch','celltype'] )
	plt.savefig('results/figure5_cnv_unique_pred_'+p1+'.png')


