import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt



df = pd.read_csv('data/figure1_umap_coordinates.csv.gz',index_col=0)


adata = sc.AnnData(df)
adata.obs['batch'] = df['batch'].astype('category')
adata.obs['celltype'] = df['celltype'].astype('category')

umap_pairs = [('c_umap1', 'c_umap2'), ('u_umap1', 'u_umap2'), ('b_umap1', 'b_umap2')]




color_palette_b = sns.color_palette("tab20", len(adata.obs['batch'].unique()))
color_palette_ct = sns.color_palette("tab20", 20)[7:]

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='batch', palette= color_palette_b,ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (Batch)")

    sc.pl.scatter(adata, x=umap_x, y=umap_y,color = 'celltype',palette=color_palette_ct, ax=axes[i, 1], show=False)
    
    axes[i, 1].set_title(f"{umap_x[0]} (Celltype)")

plt.tight_layout()
plt.show()
plt.savefig('results/figure1_latent_umaps.png')

