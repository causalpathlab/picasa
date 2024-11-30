import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd

import os
import glob

pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa'
sample = 'ovary'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
pattern = 'ovary_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

picasa_data = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	picasa_data[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break


############ read model results as adata 
wdir = pp+'/figures/'+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')


############ add metadata
dfl= pd.read_csv(ddir+sample+'_label.csv.gz')

dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]

dfl = dfl[[ 'cell', 'sample', 'patient_id',
       'treatment_phase', 'anatomical_location', 'cell_type', 'cell_subtype',
       'nCount_RNA', 'nFeature_RNA', 'percent.mt', 'celltype']]

picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')



import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype','treatment_phase'])
plt.savefig(wdir+'/results/picasa_common_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype','treatment_phase'])
plt.savefig(wdir+'/results/picasa_unique_umap.png')

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype','treatment_phase'])
plt.savefig(wdir+'/results/picasa_base_umap.png')
