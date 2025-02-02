import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

############################
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

####################################
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


picasa_adata = picasa_adata[picasa_adata.obs['celltype']!='Malignant']

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata,resolution=0.1)
sc.pl.umap(picasa_adata,color=['batch','celltype','leiden','disease'])
plt.savefig('results/figure4_cancer_unique_noncancer_umap.png')
