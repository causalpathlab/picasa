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


#### unique vs base vs common

Z_b_pseudo_inverse = np.linalg.pinv(df_b.values)
A_b = np.dot(Z_b_pseudo_inverse, df_main.values)

Z_u_pseudo_inverse = np.linalg.pinv(df_u.values)
A_u = np.dot(Z_u_pseudo_inverse, df_main.values)

Z_c_pseudo_inverse = np.linalg.pinv(df_c.values)
A_c = np.dot(Z_c_pseudo_inverse, df_main.values)



#### decomposition

from scipy.sparse.linalg import svds
_, BS, BB = svds(A_b, k = 10)
_, US, UU = svds(A_u, k = 10)
_, CS, CC = svds(A_c, k = 10)


############## 

genes = picasa_data['P1'].var.index.values

df_bb = pd.DataFrame(A_b,columns=genes)
df_uu = pd.DataFrame(A_u,columns=genes)
df_cc = pd.DataFrame(A_c,columns=genes)
# df_bb = pd.DataFrame(BB,columns=genes)
# df_uu = pd.DataFrame(UU,columns=genes)
# df_cc = pd.DataFrame(CC,columns=genes)



def top_gene_overlap(m1, m2, top_n=50):
    overlaps = []
    for i in range(m1.shape[1]):
        for j in range(m2.shape[1]):
            top_genes_1 = set(np.argsort(-np.abs(m1[:, i]))[:top_n])
            top_genes_2 = set(np.argsort(-np.abs(m2[:, j]))[:top_n])
            jaccard = len(top_genes_1 & top_genes_2) / len(top_genes_1 | top_genes_2)
            overlaps.append((i, j, jaccard))
    return pd.DataFrame(overlaps, columns=["Factor_Model1", "Factor_Model2", "Jaccard Index"])

df_bu = top_gene_overlap(df_bb.T.values, df_uu.T.values)
df_bc = top_gene_overlap(df_bb.T.values, df_cc.T.values)
df_cu = top_gene_overlap(df_cc.T.values, df_uu.T.values)

df_res = pd.DataFrame()
df_res['bu'] = df_bu['Jaccard Index']
df_res['bc'] = df_bc['Jaccard Index']
df_res['cu'] = df_cu['Jaccard Index']

df_res = pd.melt(df_res)
from plotnine import *

plot = (
    ggplot(df_res, aes(x='variable', y='value'))
    + geom_boxplot(fill="blue", alpha=0.6)
    + labs(
        title="Jaccard Index Distribution for Each Factor",
        x="Factor_Model1",
        y="Jaccard Index"
    )
    + theme(figure_size=(10, 6))
)

plot.save(wdir+'/results/unique_base_jaccard_index.png')


from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.cluster import KMeans

def eval_cluster(m1, m2,clusters):
    ari_scores = []
    for n_c in clusters:
        kmeans1 = KMeans(n_clusters=n_c, random_state=42).fit(m1)
        kmeans2 = KMeans(n_clusters=n_c, random_state=42).fit(m2)
        # ari = adjusted_rand_score(kmeans1.labels_, kmeans2.labels_)
        ari = normalized_mutual_info_score(kmeans1.labels_, kmeans2.labels_)
        ari_scores.append(ari)
    return pd.DataFrame(ari_scores, columns=['score'])


clusters = [5,10,15,20,25,30,35,40,45,50]
ari_scores_bu = eval_cluster(df_bb.T.values,df_uu.T.values,clusters)
ari_scores_bc = eval_cluster(df_bb.T.values,df_cc.T.values,clusters)
ari_scores_cu = eval_cluster(df_cc.T.values,df_uu.T.values,clusters)

df_res = pd.DataFrame()
df_res['bu'] = ari_scores_bu
df_res['bc'] = ari_scores_bc
df_res['cu'] = ari_scores_cu

df_res.reset_index(inplace=True)
df_res['index'] = pd.Categorical(df_res['index'])


from plotnine import *

df_long = df_res.melt(id_vars=["index"], var_name="variable", value_name="value")

# Create the line plot
plot = (
    ggplot(df_long, aes(x="index", y="value", color="variable"))
    + geom_point(size=1.2) 
    + labs(
        title="ARI of bu, bc, and cu",
        x="cluster increment from 5 to 50",
        y="ARI",
        color="Legend"
    )
    + theme(figure_size=(8, 5))  # Adjust the figure size
)

plot.save(wdir+'/results/unique_base_cluster_ari.png')
