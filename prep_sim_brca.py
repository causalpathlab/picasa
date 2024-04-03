
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import sailr
from scipy.sparse import csr_matrix 



############# data prep for simulation 


rna = ad.read_h5ad('data/brca/brca4290_scrna.h5ad')
spatial = ad.read_h5ad('data/brca/brca4290_spatial.h5ad')

pico = sailr.create_sailr_object({'rna':rna,'spatial':spatial})
sailr.pp.common_features(pico.data.adata_list)

rna_genes = rna.var.index.values[rna.uns['selected_genes']]



dfct = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/data/brca/brca_celltype.csv.gz')
rna.obs['celltype'] = pd.merge(rna.obs,dfct[['cell','celltype']],left_index=True, right_on='cell',how='left')['celltype'].values
selected = ['Endothelial',  'T-cells', 'Myeloid','Cancer Epithelial']
rnaf = rna[rna.obs['celltype'].isin(selected)]

rnaf = rnaf[rnaf.obs.groupby('celltype').sample(n=200).index.values]

df_rna = pd.DataFrame(rnaf.X.todense())
df_rna.index = rnaf.obs.index.values
df_rna.columns = rnaf.var.index.values

df_rna = df_rna[rna_genes]

adata_rna = ad.AnnData(X=df_rna.values)
adata_rna.obs.index = df_rna.index.values
adata_rna.var.index = df_rna.columns.values
adata_rna.obs['celltype'] = rnaf.obs['celltype'].values

adata_rna.obs['celltype'] = [x.replace(' ','') for x in adata_rna.obs['celltype'] ]
adata_rna.obs['celltype'] = [x.replace('-','') for x in adata_rna.obs['celltype'] ]
adata_rna.write('data/sim/brcasim_sc.h5ad',compression='gzip')




cell_width = 100 
cell_height = 100 
x = spatial.uns['position']['array_col'] * cell_width + spatial.uns['position']['pxl_col_in_fullres']
y = spatial.uns['position']['array_row'] * cell_height + spatial.uns['position']['pxl_row_in_fullres']
spatial.obs['position'] = [str(x)+'x'+str(y) for x,y in zip(x,y)]
spatial = spatial[spatial.obs.sample(n=278).index.values]



df_spatial = pd.DataFrame(spatial.X.todense())
df_spatial.columns = spatial.var.index.values

spatial_genes = spatial.var.index.values[spatial.uns['selected_genes']]
df_spatial = df_spatial[spatial_genes]


adata_spatial = ad.AnnData(X=df_spatial.values)
adata_spatial.obs.index = df_spatial.index.values
adata_spatial.var.index = df_spatial.columns.values


sp_ref_path = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/fig1/data/sp.h5ad'
adata_sp =  ad.read_h5ad(sp_ref_path)

adata_spatial.obs.index = adata_sp.obs['position'].values
adata_spatial.obs['position'] = adata_sp.obs['position'].values

adata_spatial.write('data/sim/brcasim_sp.h5ad',compression='gzip')

# #################################


