import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

############################
sample = 'ovary' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')
#########
####################################

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
	if batch_count >=25:
		break

picasa_data = batch_map



## Focus on cancer cells


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


batch_colors = sns.color_palette("tab20", n_colors=picasa_adata.obs['batch'].nunique()).as_hex()

treatment_phase_colors = sns.color_palette("husl", n_colors=picasa_adata.obs['treatment_phase'].nunique()).as_hex()



picasa_adata_n = picasa_adata[picasa_adata.obs['celltype']!='Malignant'].copy()
ct_cmap_n = ['royalblue',
 'lime',
 'red',
 'sienna',
 'skyblue',
 'green',
 'grey',
 'lightcoral',
 'orange']
picasa_adata_n.uns['batch_colors'] = batch_colors
picasa_adata_n.uns['celltype_colors'] = ct_cmap_n
picasa_adata_n.uns['treatment_phase_colors'] = treatment_phase_colors

sc.pp.neighbors(picasa_adata_n,use_rep='unique')
sc.tl.umap(picasa_adata_n)
sc.pl.umap(picasa_adata_n,color=['batch','celltype','treatment_phase'])
plt.savefig('results/figure4_unique_umap_subtype_noncancer.png')



picasa_adata_c = picasa_adata[picasa_adata.obs['celltype']=='Malignant'].copy()
picasa_adata_c.uns['batch_colors'] = batch_colors
picasa_adata_c.uns['celltype_colors'] = [ct_cmap_c[label] for label in picasa_adata_c.obs['celltype']]
picasa_adata_c.uns['treatment_phase_colors'] = treatment_phase_colors

sc.pp.neighbors(picasa_adata_c,use_rep='unique')
sc.tl.umap(picasa_adata_c)
sc.pl.umap(picasa_adata_c,color=['batch','celltype','treatment_phase'])
plt.savefig('results/figure4_unique_umap_subtype_cancer.png')
