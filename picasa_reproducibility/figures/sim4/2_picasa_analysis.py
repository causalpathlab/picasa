import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa'
sample = 'sim4'


############ read original data as adata list

ddir = pp+'/figures/'+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_Batch1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_Batch2.h5ad')
batch3 = an.read_h5ad(ddir+sample+'_Batch3.h5ad')

picasa_adata ={'Batch1':batch1,
	 'Batch2':batch2,
	 'Batch3':batch3
	 }

############ read model results as adata 
wdir = pp+'/figures/'+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')


############ add metadata
dfl= pd.read_csv(ddir+sample+'_label.csv.gz')
dfl = dfl[['index','Cell','batch','celltype']]
dfl.columns = ['index','cell','batch','celltype']

dfl['cell'] = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]
dfl = dfl[['index','cell','celltype']]
picasa_adata.obs = pd.merge(picasa_adata.obs,dfl,left_index=True,right_on='cell')



import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common',n_neighbors=25)
sc.tl.umap(picasa_adata,min_dist=1.0)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_common_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_unique_umap.png')

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(wdir+'/results/picasa_base_umap.png')
