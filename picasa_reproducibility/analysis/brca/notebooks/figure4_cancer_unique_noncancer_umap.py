import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

############################
sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

####################################
picasa_adata.obs['disease']= picasa_adata.obs['subtype']



picasa_adata = picasa_adata[picasa_adata.obs['celltype']!='Malignant']

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata,resolution=0.1)
sc.pl.umap(picasa_adata,color=['batch','celltype','leiden','disease'])
plt.savefig('results/figure4_cancer_unique_noncancer_umap.png')
