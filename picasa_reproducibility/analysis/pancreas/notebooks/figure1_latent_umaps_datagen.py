import scanpy as sc
import anndata as ad
import pandas as pd

############################
sample = 'pancreas' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')
####################################

df = pd.DataFrame(index=picasa_adata.obs.index)

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
df[['c_umap1','c_umap2']] = picasa_adata.obsm['X_umap']

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
df[['u_umap1','u_umap2']] = picasa_adata.obsm['X_umap']

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
df[['b_umap1','b_umap2']] = picasa_adata.obsm['X_umap']

df['batch']=picasa_adata.obs['batch']
df['celltype']=picasa_adata.obs['celltype']

df.to_csv('data/figure1_umap_coordinates.csv.gz',compression='gzip')

