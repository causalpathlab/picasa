import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

############################
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

####################################

############ read original data as adata list


ddir = wdir+'/model_data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=25:
		break

picasa_data = batch_map



## Focus on cancer cells


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


picasa_adata.obs['celltype']= [cmap[x] for x in picasa_adata.obs['celltype']]

ct_cmap_c = {
'Malignant':'magenta',
}

ct_cmap_n = {
'Monocyte':'green', 
'Plasma':'lightcoral',
'Fibroblasts':'sienna', 
'Endothelial':'red',
'T':'orange', 
'B':'royalblue',
'DC':'lime', 
'NK':'grey',
'Mast':'skyblue'
}


from picasa.util import palette
batch_colors = palette.colors_24


disease_colors = sns.color_palette("husl", n_colors=picasa_adata.obs['disease'].nunique()).as_hex()



picasa_adata_n = picasa_adata[picasa_adata.obs['celltype']!='Malignant'].copy()
ct_cmap_n = [
 'red',
 'cyan',
 'sienna',
 'green',
 'lightcoral',
 'orange',
 ]
picasa_adata_n.uns['batch_colors'] = batch_colors
picasa_adata_n.uns['celltype_colors'] = ct_cmap_n
picasa_adata_n.uns['disease_colors'] = disease_colors

sc.pp.neighbors(picasa_adata_n,use_rep='unique')
sc.tl.umap(picasa_adata_n)
sc.pl.umap(picasa_adata_n,color=['batch','celltype','disease'])
plt.savefig('results/figure3_unique_umap_subtype_noncancer.png')



picasa_adata_c = picasa_adata[picasa_adata.obs['celltype']=='Malignant'].copy()
picasa_adata_c.uns['batch_colors'] = batch_colors
picasa_adata_c.uns['celltype_colors'] = [ct_cmap_c[label] for label in picasa_adata_c.obs['celltype']]
picasa_adata_c.uns['disease_colors'] = disease_colors

sc.pp.neighbors(picasa_adata_c,use_rep='unique')
sc.tl.umap(picasa_adata_c)
sc.pl.umap(picasa_adata_c,color=['batch','celltype','disease'])
plt.savefig('results/figure3_unique_umap_subtype_cancer.png')
plt.savefig('results/figure3_unique_umap_subtype_cancer.pdf')