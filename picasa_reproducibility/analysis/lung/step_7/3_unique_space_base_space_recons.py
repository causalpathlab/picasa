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

df_main = pd.DataFrame()
for p_ad in picasa_data:
	df_main = pd.concat([df_main,picasa_data[p_ad].to_df()],axis=0)
	


#### get unique and base 

picasa_adata = ad.read_h5ad(wdir+'/fig_0/results/picasa.h5ad')
# picasa_adata = picasa_adata[picasa_adata.obs['celltype'].isin(['Malignant'])]
df_u = picasa_adata.obsm['unique']
df_u.index = ['@'.join(x.split('@')[:2])for x in df_u.index.values]
df_b = picasa_adata.obsm['base']
df_b.index = ['@'.join(x.split('@')[:2])for x in df_b.index.values]
df_c = picasa_adata.obsm['common']
df_c.index = ['@'.join(x.split('@')[:2])for x in df_c.index.values]

wdir = wdir + '/fig_7'
#### select cancer cells from main expr data
df_main = df_main.loc[df_u.index.values]


#### unique vs base

Z_b_pseudo_inverse = np.linalg.pinv(df_b.values)
A_b = np.dot(Z_b_pseudo_inverse, df_main.values)

Z_u_pseudo_inverse = np.linalg.pinv(df_u.values)
A_u = np.dot(Z_u_pseudo_inverse, df_main.values)

Z_c_pseudo_inverse = np.linalg.pinv(df_c.values)
A_c = np.dot(Z_c_pseudo_inverse, df_main.values)


def get_recons_error(df_main,A,df):
	X_b_reconstructed = df.values @ A
	b_reconstruction_error_fro = np.linalg.norm(df_main.values - X_b_reconstructed, ord='fro')
	normalized_error = b_reconstruction_error_fro / df_main.size
	return normalized_error
 


# Batch size
batch_size = 1000
b_recons = []
u_recons = []
c_recons = []

for p in picasa_adata.obs['batch'].unique():
    p_indexes = picasa_adata[picasa_adata.obs['batch']==p].obs.index.values
    p_indexes =  ['@'.join(x.split('@')[:2])for x in p_indexes]
    b_recons.append(get_recons_error(df_main.loc[p_indexes],A_b,df_b.loc[p_indexes])) 
    u_recons.append(get_recons_error(df_main.loc[p_indexes],A_u,df_u.loc[p_indexes])) 
    c_recons.append(get_recons_error(df_main.loc[p_indexes],A_c,df_c.loc[p_indexes])) 
      
    


df_recons = pd.DataFrame(b_recons,columns=['b_recons'])
df_recons['c_recons'] = c_recons
df_recons['u_recons'] = u_recons


plt.figure(figsize=(8, 6))
sns.boxplot(data=df_recons, palette="Set2", width=0.5)

for col in df_recons.columns:
    sns.stripplot(x=[col] * len(df_recons), y=df_recons[col], color='red', size=5, jitter=True, alpha=0.7)




plt.title('Reconstruction error')
plt.ylabel('MSE')
plt.xlabel('Categories')
plt.savefig(wdir+'/results/latent_recons.png')


