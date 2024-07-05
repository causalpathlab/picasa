import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


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
df.index =  [x[0].split(' ')[0] for x in barcodes.values]
df.columns = [x[0] for x in genes.values]

import picasa
hvgs = genes.values[picasa.ut.select_hvgenes(df.to_numpy(),gene_var_z=4)]
hvgs = hvgs.flatten()
print(len(hvgs))

marker = [
    "TOP2A", "AURKB", "FOXM1", "TYMS", "USP1", "EZH2", "APOD", "OLIG2",
    "STMN1", "DCX", "SOX11", "TNC", "CD44", "S100A10", "VIM", "HLA-A",
    "APOE", "HSPA1B", "DNAJB1", "HSPA6"
]

enmarker = [ x for x in df.columns if x.split('_')[1] in marker]

hvgs = np.concatenate((hvgs,np.array(enmarker)))
print(len(hvgs))

hvgs = np.unique(hvgs)
len(hvgs)


df = df.loc[:,hvgs]


adata = an.AnnData(X=df.values, obs=df.index.values, var=pd.DataFrame(index=df.columns))

adata.obs.index = adata.obs[0].values


batch = ['_'.join(x.split('_')[1:]) for x in df.index.values]

batch = [ x.replace('_1of2','').replace('_2of2','')for x in batch]
pd.Series(batch).value_counts()
adata.obs['batch'] = batch
adata.obs['celltype'] = 'unknown'

adata.obs.celltype.value_counts()
adata.obs.batch.value_counts()


# batch_keys = list(adata.obs['batch'].unique())
batch_keys = pd.Series([ x  for x in batch if 'BT' in x ]).unique()

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
    adata_b.write('gbm_'+str(batch)+'.h5ad',compression='gzip')

# dfl = adata.obs.reset_index()
# dfl = dfl.loc[:,['index','batch','celltype','celltypel3','celltypel3']]
# dfl.columns = ['cell','batch','celltype','celltypel3','celltypel3']
# dfl.to_csv('brca_label.csv.gz',compression='gzip')



import scipy.io
import pandas as pd

mat_contents = scipy.io.loadmat('annotated_cancer_data.mat')
mat_contents.keys()
cancer_barcodes = mat_contents['cancer_barcodes'].flatten()
cancer_id = mat_contents['cancer_id'].flatten()



ids = []
for arr in cancer_id: 
    for a in arr: ids.append(a)

bcodes = []
for arr in cancer_barcodes: 
    for a in arr: bcodes.append(a)

# bcodes2 = []
# for id,arrs in zip(ids,cancer_barcodes): 
#     for a in arrs: bcodes2.append(id+'_'+a)


df = pd.DataFrame()
df['barcodes'] = bcodes
df['class'] = mat_contents['cancer_class'].flatten()
df['sample'] = mat_contents['cancer_sample'].flatten()


cmap = {
0:'Unassigned',
1:'Mesenchymal',
2:'Neuronal',
3:'Astro',
4:'Progenitor',
5:'Oligo'
}

df['class'] = [cmap[x] for x in df['class']]     


smap = {}
for xi,x in enumerate(cancer_id): smap[xi+2]=x[0]
df['sample'] = [smap[x] for x in df['sample']]     
 


mat_contents = scipy.io.loadmat('gbm_sparse.mat')
mat_contents.keys()
barcodes = mat_contents['barcodes'].flatten()
cic = mat_contents['cic'].flatten()
sample = mat_contents['sample'].flatten()
clusters = mat_contents['clusters'].flatten()



bcodes = []
for arr in barcodes: 
    for a in arr: bcodes.append(a)



df2 = pd.DataFrame()
df2['barcodes'] = bcodes
df2['cic'] = cic
df2['sample'] = sample


cmap = {}
for xi,x in enumerate(clusters): cmap[xi+1]=x[0]
 
df2['cic'] = [cmap[x] for x in df2['cic']]     

df2['sample'] = [smap[x] for x in df2['sample']]  


df2.columns = ['barcodes','class','sample']

df = pd.concat([df,df2]).reset_index(drop=True)

df
# Display the combined DataFrame
print(df.head())

df.to_csv('gbm_label.csv.gz',index=False,compression='gzip')