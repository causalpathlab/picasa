import scanpy as sc
import anndata as ad
import pandas as pd

############################
sample = 'lung' 
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
'CD8T':'T', 
'Basal':'Epithelial', 
'Epithelial ':'Epithelial', 
'Alveolar':'Epithelial'
}


picasa_adata.obs['celltype']= [cmap[x] for x in picasa_adata.obs['celltype']]

pmap ={
'P10':'LUSC',
'P23':'LUSC',
'P1':'LUSC',
'P4':'LUSC',
'P37':'LUSC',
'P38':'LUAD',
'P21':'LUAD',
'P41':'LUSC',
'P25':'LUSC',
'P15':'LUSC',
'P6':'LUSC',
'P3':'LUSC',
'P39':'LUAD',
'P16':'LUAD',
'P18':'LUSC',
'P8':'LUAD',
'P28':'LUAD',
'P7':'LUSC',
'P17':'LUSC',
'P5':'LUAD',
'P14':'LUSC',
'P35':'LUAD',
'P9':'LUAD'
}

picasa_adata.obs['disease']= [pmap[x] for x in picasa_adata.obs['batch']]

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

