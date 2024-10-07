
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad


########################################

adata = ad.read_h5ad('/data/sishir/data/follym/anndata_10x_fl_all.h5ad')


adata = adata[adata.obs['timepoint'].isin(['FL','DLBCL'])]


adata.obs['timepoint'].value_counts()


adata.obs['patient'].value_counts()
adata.obs['dataset'].value_counts()
adata.obs['celltype'].value_counts()


adata.var.index = [x+'@'+y for y,x in zip(adata.var.index.values,adata.var['Symbol'].values)]

remove_cols = [ x for x in adata.var.index.values if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]

keep_cols = [ x for x in adata.var.index.values if x  not in remove_cols]
adata = adata[:,keep_cols]


sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000)
hvgs = adata.var['highly_variable'].values
print(sum(hvgs))
adata = adata[:,hvgs]



adata.obs['batch'] = adata.obs['patient']
adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()



adata = adata[adata.obs['patient'].isin([
'TFRIPAIR12',
'TFRIPAIR11',   
'TFRIPAIR8',    
'TFRIPAIR2',    
'TFRIPAIR10',
'TFRIPAIR3',
])]

adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write('follym_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.to_csv('follym_label.csv.gz',compression='gzip')


