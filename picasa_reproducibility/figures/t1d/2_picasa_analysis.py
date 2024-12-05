import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd

import os
import glob

pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa'
sample = 't1d'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
pattern = 't1d_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

picasa_data = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	picasa_data[file_name.replace('.h5ad','').replace('t1d_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break


############ read model results as adata 
wdir = pp+'/figures/'+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')


############ add metadata

adata_meta = an.read_h5ad(ddir+sample+'.h5ad')
dfl= adata_meta.obs

dfl['cell'] = [x+'@'+y for x,y in zip(dfl.index.values,dfl['disease_state'])]

dfl = dfl[['celltype', 'disease_state', 'celltype_orig', 'donor_id',
       'disease', 'assay', 'sex', 'ethnicity', 'development_stage',
       'PseudoState', 'cell']]

picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')



import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common',n_neighbors=30)
sc.tl.umap(picasa_adata,min_dist=0.3)
sc.tl.leiden(picasa_adata,resolution=0.1)
sc.pl.umap(picasa_adata,color=['batch','celltype','donor_id'])
plt.savefig(wdir+'/results/picasa_common_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='unique',n_neighbors=30)
sc.tl.umap(picasa_adata,min_dist=0.1)
sc.tl.leiden(picasa_adata,resolution=0.1)
sc.pl.umap(picasa_adata,color=['batch','celltype','donor_id'])
plt.savefig(wdir+'/results/picasa_unique_umap.png')

sc.pp.neighbors(picasa_adata,use_rep='base',n_neighbors=30)
sc.tl.umap(picasa_adata,min_dist=0.3)
sc.tl.leiden(picasa_adata,resolution=0.1)
sc.pl.umap(picasa_adata,color=['batch','celltype','donor_id'])
plt.savefig(wdir+'/results/picasa_base_umap.png')
