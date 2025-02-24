import scanpy as sc
import anndata as ad
import pandas as pd

############################
sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')
wdir = wdir + '/notebooks/'

#########

cmap = {
'Malignant':'Malignant',
'Mono/Macro':'Monocyte', 
'Plasma':'Plasma',
'Fibroblasts':'Fibroblasts', 
'Endothelial':'Endothelial',
'Tprolif':'T', 
'CD4Tconv':'T', 
'CD8Tex':'T', 
'DC':'DC', 
'Epithelial':'Epithelial', 
'B':'B',
'SMC':'SMC'
}


picasa_adata.obs['celltype']= [cmap[x] for x in picasa_adata.obs['celltype']]

dmap = {
    'CID4495': 'TNBC',
    'CID44971': 'TNBC',
    'CID4471': 'ER+',
    'CID44991': 'TNBC',
    'CID4513': 'TNBC',
    'CID3586': 'HER2+',
    'CID4066': 'HER2+',
    'CID4290A': 'ER+',
    'CID4515': 'TNBC',
    'CID4530N': 'ER+',
    'CID3963': 'TNBC',
    'CID4535': 'ER+',
    'CID4067': 'ER+',
    'CID3921': 'HER2+',
    'CID4398': 'ER+',
    'CID4040': 'ER+',
    'CID45171': 'HER2+',
    'CID3838': 'HER2+',
    'CID44041': 'TNBC',
    'CID3948': 'ER+',
    'CID4523': 'TNBC'
}

picasa_adata.obs['disease']= [dmap[x] for x in picasa_adata.obs['batch']]

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
df['disease']=picasa_adata.obs['disease']

df.to_csv(wdir+'/data/figure1_umap_coordinates.csv.gz',compression='gzip')

