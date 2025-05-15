import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


sample ='brca'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_brca.h5ad')



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


df_w = pd.DataFrame(picasa_unique_model.zinb_scale.weight.data.detach().cpu().numpy())

from scipy.stats import zscore

df_w = zscore(df_w, axis=0) 
df_w = df_w.T
df_w.columns = adata.var_names
df_w.to_csv('results/picasa_parameters_factor_by_gene.csv.gz',compression='gzip')




### patient data 

df = pd.read_csv('data/tcga_brca_expr_raw.csv.gz')
df = df.set_index('Unnamed: 0')
df.columns = [x.split('_')[1] for x in df.columns]


p = [ x for x in df_w.columns if x  in df.columns]
df = df[p]

### normalize patient data
df = np.log1p(df)
df = zscore(df, axis=1) 


#### transform to picasa space
df_w = df_w[df.columns]
df_z = df.dot(df_w.T)

df_z.columns = ['u'+str(x)for x in df_z.columns]
df_z.to_csv('results/picasa_parameters_patient_by_factor.csv.gz',compression='gzip')