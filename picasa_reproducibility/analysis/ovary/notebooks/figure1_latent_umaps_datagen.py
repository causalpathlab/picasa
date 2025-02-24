import scanpy as sc
import anndata as ad
import pandas as pd

############################
sample = 'ovary' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')
wdir = wdir + '/notebooks/'

#########

cmap = {
'EOC':'Malignant',
'Macrophages':'Monocyte', 
'Plasma':'Plasma',
'CAF':'Fibroblasts', 
'Endothelial':'Endothelial',
'T':'T', 
'B':'B',
'DC':'DC',
'NK':'NK', 
'Mast':'Mast'
}

picasa_adata.obs['celltype']= [cmap[x] for x in picasa_adata.obs['celltype']]


####################################

df = pd.DataFrame(index=picasa_adata.obs.index)

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
df[['c_umap1','c_umap2']] = picasa_adata.obsm['X_umap']
df['c_leiden'] = picasa_adata.obs['leiden']

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
df[['u_umap1','u_umap2']] = picasa_adata.obsm['X_umap']
df['u_leiden'] = picasa_adata.obs['leiden']

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
df[['b_umap1','b_umap2']] = picasa_adata.obsm['X_umap']
df['b_leiden'] = picasa_adata.obs['leiden']

df['batch']=picasa_adata.obs['batch']
df['celltype']=picasa_adata.obs['celltype']
df['treatment_phase']=picasa_adata.obs['treatment_phase']

df.to_csv(wdir+'/data/figure1_umap_coordinates.csv.gz',compression='gzip')

