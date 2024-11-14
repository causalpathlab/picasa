
import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os

sample = 'aml'
wdir = 'znode/aml/'

directory = wdir+'/data'
pattern = 'aml_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('aml_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >25:
		break


for batch_name, adata in batch_map.items():
    n_obs = adata.shape[0]

# Concatenate all AnnData objects
combined_adata = an.concat([adata for adata in batch_map.values()], merge='unique', uns_merge='unique')


adata = combined_adata


#### integration methods 


import scanpy as sc
import matplotlib.pyplot as plt
# sc.pp.filter_cells(adata, min_genes=25)
# sc.pp.filter_genes(adata, min_cells=2)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata)
# adata = adata[:, adata.var.highly_variable]
sc.tl.pca(adata,random_state=42)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)
sc.tl.umap(adata)


dfl = pd.read_csv(wdir+'data/AML_GSE116256_CellMetainfo_table.tsv',sep='\t')
# dfl.rename(columns={'index':'cell'},inplace=True)
adata.obs = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell',how='left')

plt.figure(figsize=(20, 15))
sc.pl.umap(adata, color=["Celltype (major-lineage)","Patient"])
plt.savefig(wdir+'results/scanpy_umap.png')


import bbknn
adata_bbknn = adata.copy()
sc.pp.pca(adata_bbknn)
bbknn.bbknn(adata_bbknn, batch_key='Patient')
sc.tl.leiden(adata_bbknn,resolution=1.0)
sc.tl.umap(adata_bbknn)
sc.pl.umap(adata_bbknn, color=["Celltype (major-lineage)","Patient"])
plt.savefig(wdir+'results/scanpy_umap_bbknn.png')


adata_combat = adata.copy()
sc.pp.combat(adata_combat, key='Patient')
sc.pp.scale(adata_combat)
sc.pp.pca(adata_combat, use_highly_variable=False)
sc.pp.neighbors(adata_combat)
sc.tl.umap(adata_combat)
sc.pl.umap(adata_combat, color=["Celltype (major-lineage)","Patient"])
plt.savefig(wdir+'results/scanpy_umap_combat.png')


adata_mnn = adata.copy()
adata_mnn_corrected = sc.external.pp.mnn_correct(adata_mnn, batch_key='Patient')[0][0]
sc.pp.scale(adata_mnn_corrected)
sc.pp.pca(adata_mnn_corrected)
sc.pp.neighbors(adata_mnn_corrected)  
sc.tl.leiden(adata_mnn_corrected,resolution=1.0)
sc.tl.umap(adata_mnn_corrected)
sc.pl.umap(adata_mnn_corrected, color=[ "Celltype (major-lineage)","Patient"])
plt.savefig(wdir+'results/scanpy_umap_mnn.png')


adata_harmony = adata.copy()
sc.pp.pca(adata_harmony, use_highly_variable=False)
sc.external.pp.harmony_integrate(adata_harmony, 'Patient')
sc.pp.neighbors(adata_harmony, use_rep='X_pca_harmony')
sc.tl.leiden(adata_harmony,resolution=1.0)
sc.tl.umap(adata_harmony)
sc.pl.umap(adata_harmony, color=["Celltype (major-lineage)","Patient"])
plt.savefig(wdir + 'results/scanpy_umap_harmony.png')


adata_scanorama = adata.copy()
sc.external.pp.scanorama_integrate(adata_scanorama, 'Patient')
sc.pp.neighbors(adata_scanorama,use_rep='X_scanorama')
sc.tl.leiden(adata_scanorama,resolution=1.0)
sc.tl.umap(adata_scanorama)
sc.pl.umap(adata_scanorama, color=["Celltype (major-lineage)","Patient"])
plt.savefig(wdir + 'results/scanpy_umap_scanorama.png')


## scvi -- run this in scvi-env conda

# import scvi

# adata_scvi = adata.copy()
# scvi.model.SCVI.setup_anndata(adata_scvi, batch_key="patient_id")
# model_scvi = scvi.model.SCVI(adata_scvi)
# model_scvi.view_anndata_setup()
# model_scvi.train()

# adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
# sc.pp.neighbors(adata_scvi,use_rep='X_scVI')
# sc.tl.leiden(adata_scvi,resolution=1.0)
# sc.tl.umap(adata_scvi)
# sc.pl.umap(adata_scvi, color=["patient_id", "cell_type", "treatment_phase"])
# plt.savefig(wdir + 'results/scanpy_umap_scvi.png')
