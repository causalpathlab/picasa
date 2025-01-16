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
sample = 'ovary' 
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

############ read model results as adata 
wdir = pp+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')



import scanpy as sc
import matplotlib.pylab as plt

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_common_umap_2.png')




sc.pl.umap(picasa_adata,color=['treatment_phase','celltype'])
plt.savefig(wdir+'/results/picasa_common_umap_3.png')

for b in picasa_adata.obs['batch'].unique():
    sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['treatment_phase','celltype'])
    plt.savefig(wdir+'/results/picasa_common_umap_'+b+'.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata,min_dist=0.3)
sc.pl.umap(picasa_adata[picasa_adata.obs['celltype']=='EOC',:],color=['batch','treatment_phase'])
plt.savefig(wdir+'/results/picasa_unique_umap_EOC.png')

new_adata = picasa_adata[picasa_adata.obs['celltype']=='EOC',:].copy()
sc.tl.leiden(new_adata)

new_adata.obs['leiden'].value_counts() 

new_adata2 = new_adata[new_adata.obs['leiden'].isin(['1','2','3','4','5','6']),:]
sc.pl.umap(new_adata2,color=['batch','treatment_phase'])
plt.savefig(wdir+'/results/picasa_unique_umap_EOC_top5p.png')


for b in picasa_adata.obs['batch'].unique():
    sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['treatment_phase','celltype'])
    plt.savefig(wdir+'/results/picasa_unique_umap_'+b+'.png')



sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_base_umap.png')


for b in picasa_adata.obs['batch'].unique():
    sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['treatment_phase','celltype'])
    plt.savefig(wdir+'/results/picasa_base_umap_'+b+'.png')


