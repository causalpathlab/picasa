import scanpy as sc
import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix
import anndata as an

matrix_path = 'matrix.mtx.gz'
barcodes_path = 'barcodes.tsv.gz'
genes_path = 'genes.tsv.gz'

matrix = scipy.io.mmread(matrix_path).T.tocsc()

barcodes = pd.read_csv(barcodes_path, header=None)
genes = pd.read_csv(genes_path, header=None, sep='\t')


df = pd.DataFrame(matrix.todense())
df.index = [x[0] for x in barcodes.values]
df.columns = [x[0] for x in genes.values]


remove_cols = [ x for x in df.columns if  'MT-' in x or x.split('-')[0].startswith('RPL') or x.split('-')[0].startswith('RPS') or x.split('-')[0].startswith('RP1')]

df = df[[ x for x in df.columns if x not in remove_cols]]
genes = df.columns.values

hvgs = genes[select_hvgenes(df.to_numpy(),gene_var_z=2.8)]
hvgs = hvgs.flatten()
print(len(hvgs))



marker = ['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB',
'CD4','CD1C',
'CD86','CD14',
'CXCL13', 'IL21', 'PDCD1','CCR7','IL7R',
'KLRC1', 'KLRB1', 'NKG7',
'IL1B','S100A9','FCGR3A',
'COL1A1','PDGFRA','MCAM' ]

hvgs = np.concatenate((hvgs,np.array(marker)))
print(len(hvgs))

hvgs = np.unique(hvgs)
len(hvgs)


df = df.loc[:,hvgs]


adata = an.AnnData(X=df.values, obs=df.index.to_frame(index=False), var=pd.DataFrame(index=df.columns))
adata.obs.index = adata.obs[0]


dfl = pd.read_csv('metadata.csv.gz')

dfl = dfl[['Unnamed: 0', 'orig.ident',  
    'subtype', 'celltype_subset', 'celltype_minor',
    'celltype_major']]

adata.obs['batch'] = dfl['orig.ident'].values
adata.obs['celltype'] = dfl['celltype_major'].values
adata.obs['celltypel2'] = dfl['celltype_minor'].values
adata.obs['celltypel3'] = dfl['celltype_subset'].values

adata.obs.celltype.value_counts()
dftemp = adata.obs.batch.value_counts()

### select top 8 patients with > 5k cells
sel_patients = dftemp.index.values[:8]

adata = adata[adata.obs['batch'].isin(sel_patients)]

batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df = adata_c.to_df()

    smat = csr_matrix(df.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df.columns.values
    adata_b.obs_names = df.index.values
    adata_b.obs['batch'] = adata_c.obs['batch'].values
    adata_b.obs['celltype'] = adata_c.obs['celltype'].values
    adata_b.obs['celltypel2'] = adata_c.obs['celltypel2'].values
    adata_b.obs['celltypel3'] = adata_c.obs['celltypel3'].values

    adata_b.write('brca_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs
dfl.columns = ['cell','batch','celltype','celltypel3','celltypel3']
dfl.to_csv('brca_label.csv.gz',compression='gzip')

