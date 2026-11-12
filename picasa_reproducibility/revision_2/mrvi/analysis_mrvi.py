import numpy as np
import pandas as pd 
import anndata as ad
import scanpy as sc

adata = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/data/all_pancreas.h5ad")

df_celltype = pd.read_csv("mrvi_celltype.csv", index_col=0).reindex(adata.obs_names)
df_batch = pd.read_csv("mrvi_batch.csv", index_col=0).reindex(adata.obs_names)

adata.obsm["X_mrvi_u"] = np.ascontiguousarray(df_celltype.to_numpy(), dtype=np.float32)
adata.obsm["X_mrvi_z"] = np.ascontiguousarray(df_batch.to_numpy(), dtype=np.float32)

print("Processing 'u' space (Cell Types)...")
sc.pp.neighbors(adata, use_rep="X_mrvi_u", key_added="neighbors_u")
sc.tl.umap(adata, neighbors_key="neighbors_u")
adata.obsm["X_umap_u"] = adata.obsm["X_umap"].copy() 
sc.pl.umap(adata, color=["celltype", "batch"], save="_mrvi_u.png", show=False)

print("Processing 'z' space (Batch/Samples)...")
sc.pp.neighbors(adata, use_rep="X_mrvi_z", key_added="neighbors_z")
sc.tl.umap(adata, neighbors_key="neighbors_z")
adata.obsm["X_umap_z"] = adata.obsm["X_umap"].copy() 

adata.obsm["X_umap"] = adata.obsm["X_umap_z"]
sc.pl.umap(adata, color=["celltype", "batch"], save="_mrvi_z.png", show=False)
