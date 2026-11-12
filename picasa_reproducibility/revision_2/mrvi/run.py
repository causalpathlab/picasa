import anndata as ad
import scvi
from scvi.external import MRVI
import pandas as pd

adata = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/data/all_pancreas.h5ad")
adata.layers["counts"] = adata.X

MRVI.setup_anndata(
    adata,
    layer="counts",
    sample_key="batch"
)

model = MRVI(adata)
model.train()

df_u = pd.DataFrame(model.get_latent_representation(give_z=False))
df_u.index = adata.obs.index.values
df_u.to_csv("mrvi_celltype.csv")

df_z = pd.DataFrame(model.get_latent_representation(give_z=True))
df_z.index = adata.obs.index.values
df_z.to_csv("mrvi_batch.csv")