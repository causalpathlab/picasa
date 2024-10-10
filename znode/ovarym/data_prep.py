
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


## in /data/sishir/ovarym/

files = [
    'GSM3729170_P1_dge',
    'GSM3729171_P2_dge',
    'GSM3729172_P3_dge',
    'GSM3729173_P4_dge',
    'GSM3729174_M1_dge',
    'GSM3729175_M2_dge',
    'GSM3729176_R1_dge',
    'GSM3729177_R2_dge',
    ]


common_genes = None
for f in files:
    dfc =  pd.read_csv(f+'.txt.gz',sep='\t')
    if common_genes is None:
        common_genes = set(dfc['GENE'].values)
        print(len(common_genes))
    else:
        common_genes &= set(dfc['GENE'].values)
        print(len(common_genes))
common_genes = np.array(list(common_genes))    



remove_cols = [ x for x in common_genes if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]
keep_cols = [ x for x in common_genes if x  not in remove_cols]
common_genes = keep_cols


df = pd.DataFrame() 
for f in files:
    dfc =  pd.read_csv(f+'.txt.gz',sep='\t')
    dfc = dfc.T
    dfc.columns = dfc.iloc[0,:]
    dfc = dfc.iloc[1:,:]
    dfc = dfc[common_genes]
    dfc.index = [f.split('_')[1] + '@' + x for x in dfc.index.values]
    df = pd.concat([df,dfc],axis=0)
    print(df.shape)

df = df.astype(int)






import scipy.sparse as sp
import anndata
df_sparse = sp.csr_matrix(df.values)

adata = anndata.AnnData(X=df_sparse, obs=pd.DataFrame(df.index.values), var=pd.DataFrame(df.columns.values))




sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000)
hvgs = adata.var['highly_variable'].values
print(sum(hvgs))
adata = adata[:,hvgs]


adata.obs.index = adata.obs[0].values
adata.var.index = adata.var[0].values
adata.obs['batch'] = [x.split('@')[0] for x in adata.obs.index.values]
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write('ovarym_'+str(batch)+'.h5ad',compression='gzip')


