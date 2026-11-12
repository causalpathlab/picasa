import numpy as np
import pandas as pd 
import anndata as ad
import scanpy as sc

adata_picasa = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/results/picasa.h5ad")


print("Processing 'u' space (Cell Types)...")
sc.pp.neighbors(adata_picasa, use_rep="common", key_added="neighbors_u")
sc.tl.umap(adata_picasa, neighbors_key="neighbors_u")
adata_picasa.obsm["X_umap_u"] = adata_picasa.obsm["X_umap"].copy() 
sc.pl.umap(adata_picasa, color=["celltype", "batch"], save="_picasa_u.png", show=False)

print("Processing 'z' space (Batch/Samples)...")
sc.pp.neighbors(adata_picasa, use_rep="unique", key_added="neighbors_z")
sc.tl.umap(adata_picasa, neighbors_key="neighbors_z")
adata_picasa.obsm["X_umap_z"] = adata_picasa.obsm["X_umap"].copy() 

adata_picasa.obsm["X_umap"] = adata_picasa.obsm["X_umap_z"]
sc.pl.umap(adata_picasa, color=["celltype", "batch"], save="_picasa_z.png", show=False)
