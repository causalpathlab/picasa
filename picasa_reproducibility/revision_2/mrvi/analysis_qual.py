import numpy as np
import pandas as pd 
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sc.settings.set_figure_params(dpi=300, facecolor="white", frameon=False)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"

print("Loading data...")
adata_picasa = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/results/picasa.h5ad")
adata_mrvi = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/data/all_pancreas.h5ad")

df_celltype = pd.read_csv("mrvi_celltype.csv", index_col=0).reindex(adata_mrvi.obs_names)
df_batch = pd.read_csv("mrvi_batch.csv", index_col=0).reindex(adata_mrvi.obs_names)

adata_mrvi.obsm["X_mrvi_u"] = np.ascontiguousarray(df_celltype.to_numpy(), dtype=np.float32)
adata_mrvi.obsm["X_mrvi_z"] = np.ascontiguousarray(df_batch.to_numpy(), dtype=np.float32)

for ad_obj in [adata_picasa, adata_mrvi]:
    ad_obj.obs["celltype"] = ad_obj.obs["celltype"].astype("category")
    ad_obj.obs["batch"] = ad_obj.obs["batch"].astype("category")

print("Processing PICASA latent spaces...")
sc.pp.neighbors(adata_picasa, use_rep="common", key_added="neighbors_picasa_u")
sc.tl.umap(adata_picasa, neighbors_key="neighbors_picasa_u")
adata_picasa.obsm["X_umap_picasa_u"] = adata_picasa.obsm["X_umap"].copy()

sc.pp.neighbors(adata_picasa, use_rep="unique", key_added="neighbors_picasa_z")
sc.tl.umap(adata_picasa, neighbors_key="neighbors_picasa_z")
adata_picasa.obsm["X_umap_picasa_z"] = adata_picasa.obsm["X_umap"].copy()

print("Processing MrVI latent spaces...")
sc.pp.neighbors(adata_mrvi, use_rep="X_mrvi_u", key_added="neighbors_mrvi_u")
sc.tl.umap(adata_mrvi, neighbors_key="neighbors_mrvi_u")
adata_mrvi.obsm["X_umap_mrvi_u"] = adata_mrvi.obsm["X_umap"].copy()

sc.pp.neighbors(adata_mrvi, use_rep="X_mrvi_z", key_added="neighbors_mrvi_z")
sc.tl.umap(adata_mrvi, neighbors_key="neighbors_mrvi_z")
adata_mrvi.obsm["X_umap_mrvi_z"] = adata_mrvi.obsm["X_umap"].copy()


fig, axes = plt.subplots(2, 4, figsize=(18, 8))
sub_titles = ["Cell Type", "Batch", "Cell Type", "Batch"]
for j, title in enumerate(sub_titles):
    axes[0, j].set_title(title, fontsize=11, fontweight="medium", pad=25)

adata_picasa.obsm["X_umap"] = adata_picasa.obsm["X_umap_picasa_u"]
adata_mrvi.obsm["X_umap"] = adata_mrvi.obsm["X_umap_mrvi_u"]

sc.pl.umap(adata_picasa, color="celltype", ax=axes[0, 0], show=False, legend_loc=None, title="")
sc.pl.umap(adata_picasa, color="batch", ax=axes[0, 1], show=False, palette="tab20", legend_loc=None, title="")
sc.pl.umap(adata_mrvi, color="celltype", ax=axes[0, 2], show=False, legend_loc=None, title="")
sc.pl.umap(adata_mrvi, color="batch", ax=axes[0, 3], show=False, palette="tab20", legend_loc=None, title="")

adata_picasa.obsm["X_umap"] = adata_picasa.obsm["X_umap_picasa_z"]
adata_mrvi.obsm["X_umap"] = adata_mrvi.obsm["X_umap_mrvi_z"]

sc.pl.umap(adata_picasa, color="celltype", ax=axes[1, 0], show=False, legend_loc=None, title="")
sc.pl.umap(adata_picasa, color="batch", ax=axes[1, 1], show=False, palette="tab20", legend_loc=None, title="")
sc.pl.umap(adata_mrvi, color="celltype", ax=axes[1, 2], show=False, legend_loc=None, title="")
sc.pl.umap(adata_mrvi, color="batch", ax=axes[1, 3], show=False, palette="tab20", legend_loc=None, title="")

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

fig.text(0.28, 0.94, "PICASA", ha="center", va="bottom", fontsize=14, fontweight="bold")
fig.text(0.70, 0.94, "MrVI", ha="center", va="bottom", fontsize=14, fontweight="bold")

fig.add_artist(plt.Line2D([0.08, 0.48], [0.92, 0.92], transform=fig.transFigure, color="black", linewidth=1.2))
fig.add_artist(plt.Line2D([0.50, 0.90], [0.92, 0.92], transform=fig.transFigure, color="black", linewidth=1.2))

fig.text(0.03, 0.73, "Common Space", ha="center", va="center", fontsize=12, fontweight="bold", rotation="vertical")
fig.text(0.03, 0.28, "Unique Space", ha="center", va="center", fontsize=12, fontweight="bold", rotation="vertical")

celltype_cats = adata_picasa.obs["celltype"].cat.categories
celltype_colors = adata_picasa.uns["celltype_colors"]
celltype_handles = [mpatches.Patch(color=c, label=cat) for c, cat in zip(celltype_colors, celltype_cats)]

leg1 = fig.legend(
    handles=celltype_handles, 
    title="Cell Type", 
    loc="upper left", 
    bbox_to_anchor=(0.91, 0.89), 
    frameon=False, 
    fontsize=11, 
    title_fontsize=12
)
plt.setp(leg1.get_title(), fontweight="bold")
for text in leg1.get_texts():
    text.set_fontweight("bold")

batch_cats = adata_picasa.obs["batch"].cat.categories
batch_colors = adata_picasa.uns["batch_colors"]
batch_handles = [mpatches.Patch(color=c, label=str(cat)) for c, cat in zip(batch_colors, batch_cats)]

leg2 = fig.legend(
    handles=batch_handles, 
    title="Batch", 
    loc="upper left", 
    bbox_to_anchor=(0.91, 0.44), 
    frameon=False, 
    fontsize=11, 
    title_fontsize=12
)
plt.setp(leg2.get_title(), fontweight="bold")
for text in leg2.get_texts():
    text.set_fontweight("bold")

plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.05, wspace=0.15, hspace=0.2)
output_file = "comparison_picasa_mrvi_symmetric_8panel.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()
