import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

import picasa
import anndata as an
import pandas as pd

import os
import glob 

pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'
sample = 'sim5'

############ read model results as adata 
wdir = pp+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')



import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common',n_neighbors=10)
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_common_umap.png')


for b in picasa_adata.obs['batch'].unique():
    sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['batch','celltype'])
    plt.savefig(wdir+'/results/picasa_common_umap_'+b+'.png')

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_unique_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_base_umap.png')
