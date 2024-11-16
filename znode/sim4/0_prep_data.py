
import pandas as pd
from scipy.sparse import csr_matrix
import h5py as hf
import anndata as an


wdir = "simulation_nested/"

df = pd.DataFrame()
dfl = pd.DataFrame()
for i in [1]:
    df_c= pd.read_csv(wdir+'counts_data_'+str(i)+'.csv').T
    dfl_c= pd.read_csv(wdir+'col_data_'+str(i)+'.csv')

    df = pd.concat((df,df_c),axis=0)
    dfl = pd.concat((dfl,dfl_c),axis=0)
    print(df.shape, dfl.shape)



print(dfl.groupby(['Batch','Group']).count().reset_index())
print(dfl.groupby(['Batch','Group','Condition']).count().reset_index())


smat = csr_matrix(df.to_numpy())
adata = an.AnnData(X=smat)
adata.var_names = ['g'+str(x)  for x in df.columns.values]
adata.obs_names = df.index.values


for c in dfl.columns:
    adata.obs[c] = dfl[c].values

adata.obs.index = df.index.values

adata.obs['batch'] = adata.obs['Batch'].values
adata.obs['celltype'] = adata.obs['Group'].values
adata.obs['condition'] = adata.obs['Condition'].values

print(adata.obs.celltype.value_counts())
print(adata.obs.batch.value_counts())
print(adata.obs.condition.value_counts())

wdir = ''

batch_keys = list(adata.obs['batch'].unique())

for batch in batch_keys:
    adata_c = adata[adata.obs['batch'].isin([batch])]
    df_c = adata_c.to_df()

    smat = csr_matrix(df_c.to_numpy())
    adata_b = an.AnnData(X=smat)
    adata_b.var_names = df_c.columns.values
    adata_b.obs_names = df_c.index.values

    adata_b.write(wdir+'sim4_'+str(batch)+'.h5ad',compression='gzip')

dfl = adata.obs.reset_index()
dfl.to_csv(wdir+'sim4_label.csv.gz',compression='gzip')


