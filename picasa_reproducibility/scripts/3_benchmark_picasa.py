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
picasa_adata = ad.read_h5ad(WDIR+'/results/picasa.h5ad')


WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
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

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_uc = pd.DataFrame(scaler.fit_transform(df_uc),index = idvals)
df_uc.columns = ['uc_'+str(x) for x in df_uc.columns]
df_uc.to_csv(os.path.join(RESULTS_DIR, 'benchmark_picasauc.csv.gz'),compression='gzip')

df_uc.index = picasa_adata.obs.index.values
picasa_adata.obsm['uc'] = df_uc

fig, ax = plt.subplots(figsize=(6, 6))

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasac_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasac_umap_celltype.png')

sc.pl.umap(picasa_adata,color=['batch','celltype'])
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/picasac_common_umap.pdf')

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasau_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasau_umap_celltype.png')

sc.pp.neighbors(picasa_adata,use_rep='base')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasab_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasab_umap_celltype.png')


sc.pp.neighbors(picasa_adata,use_rep='uc')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasauc_umap_batch.png')

sc.pl.umap(picasa_adata,color=['celltype'],legend_loc=None)
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('')
plt.tight_layout()
plt.savefig(RESULTS_DIR+'/scanpy_picasauc_umap_celltype.png')