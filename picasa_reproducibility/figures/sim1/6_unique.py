import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


sample ='sim1'
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

input_dim = 1000
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = 15
dec_layers = [128,128]

num_batches = len(picasa_adata.obs.batch_id.unique())

picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(nn_params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+'/results/picasa_unique.model', map_location=torch.device(nn_params['device'])))

        

p1 = 'Batch1'

adata_p1 = picasa_data[p1]

df_h = pd.DataFrame()
df_zc = pd.DataFrame()
df_zu = pd.DataFrame()

cn = 2750
for i in range(cn):
    x_gene = adata_p1.X[0].toarray().T
    h,zc,zu = picasa_unique_model.get_common_unique_representation(x_gene)
    df_h = pd.concat([df_h,pd.DataFrame(h.detach().numpy())])
    df_zc = pd.concat([df_zc,pd.DataFrame(zc.detach().numpy())])
    df_zu = pd.concat([df_zu,pd.DataFrame(zu.detach().numpy())])


new_adata = adata_p1[:cn,:].copy()
new_adata.obsm['X_mixed'] = df_h.values
new_adata.obsm['X_common'] = df_zc.values
new_adata.obsm['X_unique'] = df_zu.values

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.neighbors(new_adata, use_rep="X_mixed")
sc.tl.umap(new_adata)
sc.pl.umap(new_adata,color=['batch','celltype'] )
plt.savefig(wdir+'/results/picasa_unique_proj_mix.png')

sc.pp.neighbors(new_adata, use_rep="X_common")
sc.tl.umap(new_adata)
sc.pl.umap(new_adata,color=['batch','celltype'] )
plt.savefig(wdir+'/results/picasa_unique_proj_c.png')

sc.pp.neighbors(new_adata, use_rep="X_unique")
sc.tl.umap(new_adata)
sc.pl.umap(new_adata,color=['batch','celltype'] )
plt.savefig(wdir+'/results/picasa_unique_proj_u.png')