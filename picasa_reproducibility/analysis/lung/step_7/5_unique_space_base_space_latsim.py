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
picasa_adata = picasa_adata[picasa_adata.obs['celltype'].isin(['Malignant'])]
df_u = picasa_adata.obsm['unique']
df_u.index = ['@'.join(x.split('@')[:2])for x in df_u.index.values]
df_c = picasa_adata.obsm['common']
df_c.index = ['@'.join(x.split('@')[:2])for x in df_c.index.values]
df_b = picasa_adata.obsm['base']
df_b.index = ['@'.join(x.split('@')[:2])for x in df_b.index.values]

wdir = wdir + '/fig_7'
#### select cancer cells from main expr data
df_main = df_main.loc[df_u.index.values]


#### unique vs base

Z_b_pseudo_inverse = np.linalg.pinv(df_b.values)
A_b = np.dot(Z_b_pseudo_inverse, df_main.values)
X_b_reconstructed = np.dot(df_b.values, A_b)
b_reconstruction_error_fro = np.linalg.norm(df_main.values - X_b_reconstructed, ord='fro')

Z_u_pseudo_inverse = np.linalg.pinv(df_u.values)
A_u = np.dot(Z_u_pseudo_inverse, df_main.values)
X_u_reconstructed = np.dot(df_u.values, A_u)
u_reconstruction_error_fro = np.linalg.norm(df_main.values - X_u_reconstructed, ord='fro')

Z_c_pseudo_inverse = np.linalg.pinv(df_c.values)
A_c = np.dot(Z_c_pseudo_inverse, df_main.values)
X_c_reconstructed = np.dot(df_c.values, A_c)
c_reconstruction_error_fro = np.linalg.norm(df_main.values - X_c_reconstructed, ord='fro')




#### decomposition

from scipy.sparse.linalg import svds
_, BS, BB = svds(A_b, k = 24)
_, US, UU = svds(A_u, k = 24)
_, CS, CC = svds(A_c, k = 24)


############## reconstruction
##############

df_bb = pd.DataFrame(BB,columns=picasa_data['P1'].var.index.values)
df_uu = pd.DataFrame(UU,columns=picasa_data['P1'].var.index.values)
df_cc = pd.DataFrame(CC,columns=picasa_data['P1'].var.index.values)


from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def get_sim(df,dfl):
	similarities = []
	for v1 in range(df.shape[0]):
		for v2 in range(dfl.shape[0]):
			col_main = df.iloc[v1, :].values.reshape(1, -1)
			col_uu = dfl.iloc[v2, :].values.reshape(1, -1)
			
			coss = cosine_similarity(col_uu, col_main)[0][0]
			similarities.append(coss)
	return similarities


def get_corr(df, dfl):
	correlations = []
	for v1 in range(df.shape[0]):
		for v2 in range(dfl.shape[0]):
			col_main = df.iloc[v1, :].values
			col_uu = dfl.iloc[v2, :].values
			
			corr, _ = pearsonr(col_uu, col_main)
			correlations.append(corr)
		
	return correlations

b_u_sim = get_corr(df_bb,df_uu)
b_c_sim = get_corr(df_bb,df_cc)
u_c_sim = get_corr(df_uu,df_cc)

df_sim = pd.DataFrame(b_u_sim,columns=['bu'])
df_sim['bc'] = b_c_sim
df_sim['uc'] = u_c_sim

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_sim, palette="Set2")

for col in df_sim.columns:
	sns.stripplot(x=[col] * len(df_sim), y=df_sim[col], color='red', size=2, jitter=True, alpha=0.7)




plt.title('Similarity measure')
plt.ylabel('Mean Value')
plt.xlabel('Categories')
plt.savefig(wdir+'/results/latent_corr_sim.png')



b_u_sim = get_sim(df_bb,df_uu)
b_c_sim = get_sim(df_bb,df_cc)
u_c_sim = get_sim(df_uu,df_cc)

df_sim = pd.DataFrame(b_u_sim,columns=['bu'])
df_sim['bc'] = b_c_sim
df_sim['uc'] = u_c_sim


plt.figure(figsize=(8, 6))
sns.boxplot(data=df_sim, palette="Set2")

for col in df_sim.columns:
	sns.stripplot(x=[col] * len(df_sim), y=df_sim[col], color='red', size=2, jitter=True, alpha=0.7)

plt.title('Similarity measure')
plt.ylabel('Mean Value')
plt.xlabel('Categories')
plt.savefig(wdir+'/results/latent_cosine_sim.png')
