import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

from scipy.sparse import csr_matrix

import h5py as hf
import anndata as an
adata = an.read_h5ad('gbm.h5ad')



sel_donors = [
    "BT333",
    "BT346",
    "BT363",
    "BT364",
    "BT368",
    "BT389",
    "BT390",
    "BT397",
    "BT402",
    "BT407",
    "BT409"
]

sel_donors = [
    'PJ017', 'PJ018', 'PJ025', 'PJ032', 'PJ035', 'PJ048']


adata = adata[adata.obs['donor_id'].isin(sel_donors)]

df = adata.to_df()
df['cell_type'] = adata.obs['cell_type']

def sample_n(group, n):
    if len(group) < n:
        return group
    return group.sample(n=n, random_state=42)

n = 1000
df_sampled = df.groupby('cell_type', group_keys=False).apply(sample_n, n=n)
df_sampled['cell_type'].value_counts()

adata = adata[df_sampled.index.values]

adata.obs['donor_id'].value_counts()
adata.obs['cell_type'].value_counts()

genes = adata.var['feature_name'].values

import picasa
import numpy as np
hvgs = genes[picasa.ut.select_hvgenes(adata.X.toarray(),gene_var_z=1.5)]
print(len(hvgs))

marker = [
    "TOP2A", "AURKB", "FOXM1", "TYMS", "USP1", "EZH2", "APOD", "OLIG2",
    "STMN1", "DCX", "SOX11", "TNC", "CD44", "S100A10", "VIM", "HLA-A",
    "APOE", "HSPA1B", "DNAJB1", "HSPA6"
]

enmarker = [ x for x in marker if x in genes]

hvgs = np.concatenate((hvgs,np.array(marker)))
print(len(hvgs))

hvgs = np.unique(hvgs)
len(hvgs)

adata = adata[:,adata.var['feature_name'].isin(hvgs)]



adata.obs['batch'] = adata.obs['donor_id']
adata.obs['celltype'] = adata.obs['cell_type']

adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    print(batch)
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values
    adata_b.write('gbm2_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl = dfl.loc[:,['index','donor_id','cell_type']]
dfl.columns = ['cell','batch','celltype']
dfl.to_csv('gbm2_label.csv.gz',compression='gzip')


