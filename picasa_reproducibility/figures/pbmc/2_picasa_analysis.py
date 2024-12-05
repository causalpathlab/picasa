import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa'
sample = 'pbmc'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_pbmc1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_pbmc2.h5ad')
picasa_data = {'pbmc1':batch1,'pbmc2':batch2}


############ read model results as adata 
wdir = pp+'/figures/'+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')


############ add metadata
dfl= pd.read_csv(ddir+sample+'_label.csv.gz')
dfl.columns = ['index','cell','batch','celltype']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]
dfl = dfl[['index','cell','celltype']]
picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')



import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_common_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
picasa_adata.obsp['unique_distances'] = picasa_adata.obsp['distances']
picasa_adata.obsp['unique_connectivities'] = picasa_adata.obsp['connectivities']
del picasa_adata.obsp['distances']
del picasa_adata.obsp['connectivities']
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_unique_umap.png')

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
picasa_adata.obsp['base_distances'] = picasa_adata.obsp['distances']
picasa_adata.obsp['base_connectivities'] = picasa_adata.obsp['connectivities']
del picasa_adata.obsp['distances']
del picasa_adata.obsp['connectivities']
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_base_umap.png')
