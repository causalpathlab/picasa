import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
# sample = sys.argv[1] 
sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')

####################################

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_common_umap.png')



# for b in picasa_adata.obs['batch'].unique():
#     sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['celltype'])
#     plt.title(b)
#     plt.tight_layout()
#     plt.savefig(wdir+'/results/picasa_common_umap_'+b+'.png')


sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_unique_umap.png')


# for b in picasa_adata.obs['batch'].unique():
#     sc.pl.umap(picasa_adata[picasa_adata.obs['batch']==b],color=['celltype'])
#     plt.title(b)
#     plt.tight_layout()
#     plt.savefig(wdir+'/results/picasa_unique_umap_'+b+'.png')



sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(wdir+'/results/picasa_base_umap.png')



