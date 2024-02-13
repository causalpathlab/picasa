
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd

import picasa

rna = ad.read_h5ad('data/brca4290_scrna.h5ad')
spatial = ad.read_h5ad('data/brca4290_spatial.h5ad')


pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})

picasa.pp.common_features(pico.data.adata_list) 

picasa.proj.rp(pico.data.adata_list)


df = pd.read_csv('data/brca_celltype.csv.gz')
rna.obs['celltype'] = pd.merge(rna.obs,df,left_index=True,right_on='cell',how='left')['celltype'].values

sc.pp.neighbors(rna,n_neighbors=10,use_rep='X_rp')
sc.tl.umap(rna)
sc.tl.leiden(rna)
sc.pl.umap(rna,color=['celltype'])
plt.savefig('data/test.png');plt.close()

sc.pp.neighbors(spatial,n_neighbors=10,use_rep='X_rp')
sc.tl.umap(spatial)
sc.tl.leiden(spatial)
sc.pl.umap(rna,color=['leiden'])
plt.savefig('data/test2.png');plt.close()

