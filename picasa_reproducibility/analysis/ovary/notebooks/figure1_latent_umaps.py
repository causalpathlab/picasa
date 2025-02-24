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
adata.obs['treatment_phase'] = df['treatment_phase'].astype('category')
adata.obs['c_leiden'] = df['c_leiden'].astype('category')
adata.obs['u_leiden'] = df['u_leiden'].astype('category')
adata.obs['b_leiden'] = df['b_leiden'].astype('category')

umap_pairs = [('c_umap1', 'c_umap2'), ('u_umap1', 'u_umap2'), ('b_umap1', 'b_umap2')]


cmap = {
'Malignant':'magenta',
'Monocyte':'green', 
'Plasma':'lightcoral',
'Fibroblasts':'sienna', 
'Endothelial':'red',
'T':'orange', 
'B':'royalblue',
'DC':'lime', 
'NK':'grey',
'Mast':'skyblue'
}

color_palette = sns.color_palette("tab20", len(adata.obs['batch'].unique()))

cust_palette = [cmap[label] for label in adata.obs['celltype'].cat.categories]


fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='batch', palette= color_palette,ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (Batch)")

    sc.pl.scatter(adata, x=umap_x, y=umap_y,color = 'celltype',palette=cust_palette, ax=axes[i, 1], show=False)
    
    axes[i, 1].set_title(f"{umap_x[0]} (Celltype)")

plt.tight_layout()
plt.savefig('results/figure1_latent_umaps.png')


fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='batch', palette= color_palette,ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (Batch)")

    sc.pl.scatter(adata, x=umap_x, y=umap_y,color = 'celltype',palette=cust_palette, ax=axes[i, 1], show=False)
    
    axes[i, 1].set_title(f"{umap_x[0]} (Celltype)")
    
    break

plt.savefig('results/figure1_latent_umaps.pdf')


fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='batch', ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (Batch)")

    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='treatment_phase', ax=axes[i, 1], show=False)
    axes[i, 1].set_title(f"{umap_x[0]} (treatment_phase)")

plt.tight_layout()
plt.show()
plt.savefig('results/figure1_latent_umaps_disease.png')

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y,color = 'celltype',palette=cust_palette, ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (celltype)")

    tag = umap_x.split('_')[0]
    tag = tag+'_leiden'
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color=tag, ax=axes[i, 1], show=False)
    axes[i, 1].set_title(f"{umap_x[0]} (Leiden)")

plt.tight_layout()
plt.show()
plt.savefig('results/figure1_latent_umaps_leiden.png')


