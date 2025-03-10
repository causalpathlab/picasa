import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
sample = sys.argv[1] 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/test'

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')

####################################

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_umap_batch.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_unique_umap.png')


sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_base_umap.png')



