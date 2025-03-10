import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
sample = sys.argv[1] 
# sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

####################################

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
plt.tight_layout()
plt.savefig(wdir+'/benchmark_results/scanpy_picasa_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(wdir+'/benchmark_results/scanpy_picasa_umap_celltype.png')

sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(wdir+'/benchmark_results/picasa_common_umap.pdf')

