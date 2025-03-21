import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
import os
import pandas as pd


############################
SAMPLE = sys.argv[1] 
# SAMPLE = 'sim2' 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+SAMPLE

############ read model results as adata 
picasa_adata = ad.read_h5ad(WDIR+'/model_results/picasa.h5ad')


WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'model_data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')

####################################

if SAMPLE == 'lung':
    idvals = ['@'.join(x.split('@')[:2])for x in picasa_adata.obsm['base'].index.values]
    
else: 
    idvals = [x.split('@')[0] for x in picasa_adata.obsm['base'].index.values]


pd.DataFrame(picasa_adata.obsm['base'].values,index=idvals).to_csv(os.path.join(RESULTS_DIR, 'benchmark_picasab.csv.gz'),compression='gzip')

df_u = pd.DataFrame(picasa_adata.obsm['unique'].values,index=idvals)
df_u.to_csv(os.path.join(RESULTS_DIR, 'benchmark_picasau.csv.gz'),compression='gzip')

df_c = pd.DataFrame(picasa_adata.obsm['common'].values,index=idvals)
df_c.to_csv(os.path.join(RESULTS_DIR, 'benchmark_picasac.csv.gz'),compression='gzip')

df_uc = pd.concat([df_u,df_c],axis=1)
df_uc.to_csv(os.path.join(RESULTS_DIR, 'benchmark_picasauc.csv.gz'),compression='gzip')


sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasac_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasac_umap_celltype.png')

sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/picasac_common_umap.pdf')

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasau_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasau_umap_celltype.png')

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasab_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasab_umap_celltype.png')