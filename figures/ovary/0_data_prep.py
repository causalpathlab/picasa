
import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


########### read raw from web and generate main adata

dfl = pd.read_csv('GSE165897_cellInfo_HGSOC.tsv.gz',sep='\t')

df = pd.read_csv('GSE165897_UMIcounts_HGSOC.tsv.gz',sep='\t')
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]


adata = an.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))

for c in dfl.columns:
    adata.obs[c] = dfl[c].values


# t = [1 if x ==y else 0 for x,y in zip(adata.obs.index.values,adata.obs['cell'].values)]

adata.X = adata.X.astype(int)
adata.write('ovary_main.h5ad',compression='gzip')

########################################



adata = an.read_h5ad('ovary_main.h5ad')


remove_cols = [ x for x in adata.var.index.values if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]
keep_cols = [ x for x in adata.var.index.values if x  not in remove_cols]
adata = adata[:,keep_cols]

# adata = adata[adata.obs['treatment_phase']=='treatment-naive']



sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000)
hvgs = adata.var['highly_variable'].values
print(sum(hvgs))
adata = adata[:,hvgs]



adata.obs['batch'] = adata.obs['patient_id']
adata.obs['celltype'] = adata.obs['cell_type']
adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write('ovary_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.to_csv('ovary_label.csv.gz',compression='gzip')


