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

wdir = wdir + '/fig_7'
#### select cancer cells from main expr data
df_main = df_main.loc[df_u.index.values]


#### unique vs base
Z_b_pseudo_inverse = np.linalg.pinv(df_b.values)
A_b = np.dot(Z_b_pseudo_inverse, df_main.values)
X_b_reconstructed = np.dot(df_b.values, A_b)
b_reconstruction_error = np.mean(np.abs(df_main.values - X_b_reconstructed))

Z_u_pseudo_inverse = np.linalg.pinv(df_u.values)
A_u = np.dot(Z_u_pseudo_inverse, df_main.values)
X_u_reconstructed = np.dot(df_u.values, A_u)
u_reconstruction_error = np.mean(np.abs(df_main.values - X_u_reconstructed))


# from scipy.spatial.distance import cdist
# correlation_distance = cdist(A_b.T, A_u.T, metric="correlation")
# correlation_matrix = 1 - correlation_distance

# num_rows = correlation_matrix.shape[0]
# sampled_indices = np.random.choice(num_rows, size=100, replace=False)
# sampled_rows = correlation_matrix[sampled_indices, :]

# sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
# plt.title("Correlation Between Latent Factors")
# plt.savefig(wdir+'/results/unique_base_corr_genes.png')


#### decomposition

from scipy.sparse.linalg import svds
_, BS, BB = svds(A_b, k = 15)
_, US, UU = svds(A_u, k = 15)


############## reconstruction
be = df_main.values @ BB.T @ BB
ue = df_main.values @ UU.T @ UU

dfl = picasa_adata.obs.copy()
dfl.index = ['@'.join(x.split('@')[:2])for x in dfl.index.values]
dfl = dfl.loc[df_main.index.values]

adata_be = ad.AnnData(be)
adata_be.obs = dfl
sc.pp.pca(adata_be)
sc.pp.neighbors(adata_be)
sc.tl.umap(adata_be)
sc.pl.umap(adata_be,color=['batch','celltype'])
plt.savefig(wdir+'/results/umap_base_effect.png')

adata_ue = ad.AnnData(ue)
adata_ue.obs = dfl
sc.pp.pca(adata_ue)
sc.pp.neighbors(adata_ue)
sc.tl.umap(adata_ue)
sc.pl.umap(adata_ue,color=['batch','celltype'])
plt.savefig(wdir+'/results/umap_unique_effect.png')



##############

df_bb = pd.DataFrame(BB,columns=picasa_data['P1'].var.index.values)
df_uu = pd.DataFrame(UU,columns=picasa_data['P1'].var.index.values)

adata_be = ad.AnnData(df_bb.T)
adata_be.obs.index = picasa_data['P1'].var.index.values
sc.pp.pca(adata_be)
sc.pp.neighbors(adata_be)
sc.tl.umap(adata_be)
sc.tl.leiden(adata_be)

adata_ue = ad.AnnData(df_uu.T)
adata_ue.obs.index = picasa_data['P1'].var.index.values
sc.pp.pca(adata_ue)
sc.pp.neighbors(adata_ue)
sc.tl.umap(adata_ue)
sc.tl.leiden(adata_ue)

df_b_u = adata_be.obs.copy()
df_b_u['base'] = df_b_u['leiden'] 
df_b_u['unique'] = adata_ue.obs['leiden'] 
df_b_u.reset_index(inplace=True)


df_b_u['random'] = np.random.randint(0, 15, size=2020)

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ami = adjusted_mutual_info_score(df_b_u['base'], df_b_u['unique'])
ari = adjusted_mutual_info_score(df_b_u['base'], df_b_u['unique'])

print(f"Adjusted Mutual Information Score: {ami}")
print(f"Adjusted Rand Index: {ari}")

ami = adjusted_mutual_info_score(df_b_u['base'], df_b_u['random'])
ari = adjusted_mutual_info_score(df_b_u['base'], df_b_u['random'])


print(f"Adjusted Mutual Information Score: {ami}")
print(f"Adjusted Rand Index: {ari}")