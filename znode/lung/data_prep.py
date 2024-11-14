
import anndata as an
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt

import h5py as hf
f = hf.File('NSCLC_GSE148071_expression.h5')

mtx_indptr = f['matrix']['indptr']
mtx_indices = f['matrix']['indices']
mtx_data = f['matrix']['data']
barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
features = [x.decode('utf-8') for x in f['matrix']['features']['id']]

n_genes = len(features)
n_cells = len(barcodes)

matrix = csr_matrix((mtx_data, mtx_indices, mtx_indptr), shape=(n_cells, n_genes))

adata = an.AnnData(X=matrix)
adata.obs.index = barcodes
adata.var.index = features

remove_cols = [ x for x in features if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]
keep_cols = [ x for x in features if x  not in remove_cols]

adata[:,keep_cols]


sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000)
hvgs = adata.var['highly_variable'].values
print(sum(hvgs))
adata = adata[:,hvgs]

dfl = pd.read_csv('NSCLC_GSE148071_CellMetainfo_table.tsv',sep='\t')

# patients more than 2.5k cells
sel_patient = dfl.Patient.value_counts().index[:11]
dfl = dfl[dfl['Patient'].isin(sel_patient)]

adata = adata[dfl['Cell'].values,:]
adata.obs['batch'] = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell')['Patient'].values

adata.obs['batch'].value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write('lung_'+str(batch)+'.h5ad',compression='gzip')


