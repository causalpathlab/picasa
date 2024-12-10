
import pandas as pd
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc 

import scanpy as sc

# List of file paths
file_list = [
    "sim6_Batch1.h5ad",
    "sim6_Batch2.h5ad",
    "sim6_Batch3.h5ad",
    "sim6_Batch4.h5ad"
]

# Initialize an empty list to store AnnData objects
adata_list = []

# Loop through each file and load it
for file in file_list:
    adata = sc.read_h5ad(file)  # Read the AnnData file
    adata_list.append(adata)   # Append to the list

# Concatenate all AnnData objects into one
merged_adata = an.concat(adata_list, axis=0)

adata = merged_adata



adata = adata[adata.obs['celltype'] != 'Group5']
adata = adata[adata.obs['celltype'] != 'Group6']
adata = adata[adata.obs['celltype'] != 'Group7']

print(adata.obs.celltype.value_counts())
print(adata.obs.batch.value_counts())



wdir = ''

batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values

    adata_b.write(wdir+'sim6_'+str(batch)+'.h5ad',compression='gzip')


