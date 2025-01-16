import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

import picasa
import anndata as an
import pandas as pd

import os
import glob 

# sample = sys.argv[1] 
# pp = sys.argv[2]
sample = 'lung' 
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

############ read model results as adata 
wdir = pp+sample
picasa_adata = an.read_h5ad(wdir+'/fig_0/results/picasa.h5ad')
wdir = wdir +'/fig_7/'


import scanpy as sc
import matplotlib.pylab as plt


picasa_adata = picasa_adata[picasa_adata.obs['celltype'].isin(['Malignant'])]
sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata,0.5)

picasa_adata.obs['leiden'].value_counts()

###select clusters with >1k cells
# label_counts = picasa_adata.obs['leiden'].value_counts()
# filtered_labels = label_counts.index.values[:10]
# picasa_adata = picasa_adata[picasa_adata.obs['leiden'].isin(filtered_labels)]

sc.pl.umap(picasa_adata,color=['batch','celltype','leiden'])
plt.savefig(wdir+'/results/picasa_unique_patient.png')

picasa_adata.obs.to_csv(wdir+'/results/unique_space_selected_cells.csv.gz',compression='gzip')

