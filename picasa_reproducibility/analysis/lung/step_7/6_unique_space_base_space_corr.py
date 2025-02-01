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
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

picasa_adata = picasa_adata[picasa_adata.obs['celltype'].isin(['Malignant'])]


df_u = picasa_adata.obsm['unique']
df_b = picasa_adata.obsm['base']


############### check correlation
correlation_distance = cdist(df_u, df_b, metric="correlation")
correlation_matrix = 1 - correlation_distance
num_rows = correlation_matrix.shape[0]
sampled_indices = np.random.choice(num_rows, size=100, replace=False)
sampled_rows = correlation_matrix[sampled_indices, :]

sns.heatmap(sampled_rows, annot=False, cmap="coolwarm")
plt.title("Correlation Between Latent Factors")
plt.savefig(wdir+'/results/unique_base_corr_patients.png')




####################