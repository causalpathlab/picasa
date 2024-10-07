
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

## train 
adata = an.read_h5ad('ovary_main.h5ad')
train_genes = adata.var.index.values

##test
files = [
    'GSM3729170_P1_dge',
    'GSM3729171_P2_dge',
    'GSM3729172_P3_dge',
    'GSM3729173_P4_dge'
    ]
fp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/znode/ovary_ext_p/test_data/'
test_genes = None
for f in files:
    dfc =  pd.read_csv(fp+f+'.txt.gz',sep='\t')
    if test_genes is None:
        test_genes = set(dfc['GENE'].values)
        print(len(test_genes))
    else:
        test_genes &= set(dfc['GENE'].values)
        print(len(test_genes))
test_genes = np.array(list(test_genes))    

common_genes = np.intersect1d(train_genes,test_genes)

remove_genes = [ x for x in common_genes if  'MT-' in x or x.startswith('RPL') or x.startswith('RPS') or x.startswith('RP1') or x.startswith('MRP')]
keep_genes = [ x for x in common_genes if x  not in remove_genes]

adata = adata[:,adata.var.index.isin(keep_genes)]

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


