import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as ad
import pandas as pd


sample ='lung'
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
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

picasa_data = batch_map


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')



from picasa import model,dutil
import torch

nn_params = picasa_adata.uns['nn_params']
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/results/picasa_common.model', map_location=torch.device(nn_params['device'])))


picasa_common_model.eval()

p1 = 'P6'
adata_p1 = picasa_data[p1]

# df_z = pd.DataFrame()
# batch_size = 100
# for start in range(0, adata_p1.shape[0], batch_size):
#     end = min(start + batch_size, adata_p1.shape[0])
#     x_gene = adata_p1.X[start:end].toarray()
#     x_gene = torch.tensor(x_gene).float()
#     picasa_out = picasa_common_model.estimate(x_gene)
#     df_z = pd.concat([df_z,pd.DataFrame(picasa_out.h_c1.detach().numpy())])


# new_adata = adata_p1.copy()
# new_adata.obsm['X_common'] = df_z.values

# import scanpy as sc
# import matplotlib.pyplot as plt
# sc.pp.neighbors(new_adata, use_rep="X_common")
# sc.tl.umap(new_adata)
# sc.pl.umap(new_adata,color=['batch','celltype'] )
# plt.savefig(wdir+'/results/picasa_unique_proj_c.png')



### patient data 

df = pd.read_csv(pp+sample+'/data/tcga_lung_expr_raw.csv.gz')
df = df.set_index('Unnamed: 0')
df.columns = [x.split('_')[1] for x in df.columns]
p = [ x for x in adata_p1.var_names if x  in df.columns]
np = [ x for x in adata_p1.var_names if x not in df.columns]
df = df[p]
for col in np:df[col] = 0.0
df = df.loc[:,adata_p1.var_names]


df = df.div(df.sum(axis=1), axis=0) * 10000

df_z = pd.DataFrame()
batch_size = 100
for start in range(0, df.shape[0], batch_size):
    end = min(start + batch_size, df.shape[0])
    x_gene = df.iloc[start:end,:].values
    x_gene = torch.tensor(x_gene).float()
    picasa_out = picasa_common_model.estimate(x_gene)
    df_z = pd.concat([df_z,pd.DataFrame(picasa_out.h_c1.detach().numpy())])


new_adata = ad.AnnData(X=df.values)
new_adata.var_names = df.columns
new_adata.obs_names = df.index
new_adata.obsm['X_common'] = df_z.values

df = pd.read_csv(pp+sample+'/data/tcga_lung_clinical.csv.gz')

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.neighbors(new_adata, use_rep="X_common")
sc.tl.umap(new_adata)
sc.tl.leiden(new_adata)
sc.pl.umap(new_adata,color=['leiden'] )
plt.savefig(wdir+'/results/picasa_unique_proj_c_patient.png')



nn_params = picasa_adata.uns['nn_params']
input_dim = nn_params['input_dim']
enc_layers = [128,25]
unique_latent_dim = nn_params['latent_dim']
common_latent_dim = nn_params['latent_dim']
dec_layers = [128,128]
num_batches = len(picasa_adata.obs.batch_id.unique())

picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(nn_params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+'/results/picasa_unique.model', map_location=torch.device(nn_params['device'])))

picasa_unique_model.eval()


x_c1 = torch.tensor(new_adata.X).float()
x_z = torch.tensor(new_adata.obsm['X_common']).float()
y = new_adata.obs_names.values

z,ylabel = model.predict_batch_unique(picasa_unique_model,x_c1,y,x_z)
z_u = z[0]

new_adata.obsm['X_unique'] = z_u.detach().numpy()

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.neighbors(new_adata, use_rep="X_unique")
sc.tl.umap(new_adata)
sc.tl.leiden(new_adata)
sc.pl.umap(new_adata,color=['leiden'] )
plt.savefig(wdir+'/results/picasa_unique_proj_u_patient.png')


new_adata.write_h5ad(wdir+'/results/picasa_proj_u_patient.h5ad')

