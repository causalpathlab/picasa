import scanpy as sc
import infercnvpy as cnv
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from plotnine import * 
import sys


sample ='brca'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')
df_obs = picasa_adata.obs.copy()
df_obs.index = [x.split('@')[0] for x in df_obs.index.values]


# adata = cnv.datasets.maynard2020_3k()
# adata.var.loc[:, ["ensg", "chromosome", "start", "end"]]
# adata.var.to_csv(wdir+'/model_data/df_gene_coord.csv.gz')
df_gene = pd.read_csv(wdir+'/model_data/df_gene_coord.csv.gz',index_col=0)



def sample_or_take_all(group, n):
	return group.sample(n=min(len(group), n), random_state=42)

def cnv_analysis(df_expr,df_gene,df_obs,tag):
	

	adata = ad.AnnData(df_expr)
	adata.obs_names = df_expr.index.values
	adata.var_names = df_expr.columns.values
	adata.var = df_gene
	adata.obs = df_obs

	cats = adata.obs.batch.unique().tolist()


	cnv.tl.infercnv(
		adata,
		reference_key="batch",
		reference_cat = cats,
    	window_size=250,
    	step=1	
     )



	plt.xlabel("chr", fontsize=14)
	plt.ylabel("patient", fontsize=14)
	plt.rcParams.update({'font.size': 14})
 	
	df_cnv = pd.DataFrame(adata.obsm['X_cnv'].todense())
	df_cnv.index = adata.obs.index.values
	df_cnv.to_parquet(f'results/figure3_cnv_{tag}_profile.parquet', engine='pyarrow', compression='snappy')

	adata.obsm['X_cnv'].data = np.clip(adata.obsm['X_cnv'].data, -0.25, 0.25)

	cnv.pl.chromosome_heatmap(adata, 
		figsize=(25, 20),
		groupby="batch")
		
	plt.savefig('results/figure3_cnv_'+tag+'.pdf')


	# cnv.tl.pca(adata)
	# cnv.pp.neighbors(adata,n_neighbors=30)
	# cnv.tl.leiden(adata,resolution=0.5)
	# cnv.tl.umap(adata,min_dist=1)
	# cnv.pl.umap(adata, color=['cnv_leiden','celltype','batch'])

	# plt.savefig('results/figure3_cnv_umap_'+tag+'.png')

	# df_cnv = pd.DataFrame(adata.obsm['X_cnv'].todense())
	# df_cnv.to_csv('results/cnv_'+tag+'.csv.gz',compression='gzip')

	# plot_prop(adata.obs.copy(),tag)


tag = 'orig'
adata = ad.read_h5ad(wdir+'/model_data/all_brca.h5ad')
df_expr = adata.to_df() ## 72k x 25k

present_genes = [ x for x in df_expr.columns if x  in df_gene.index.values]
## 24k genes found to have coordiante

df_gene = df_gene.loc[present_genes]
df_expr = df_expr.loc[:,present_genes]
df_obs = df_obs.loc[df_expr.index.values]


### sample 1000 cells
n=1000
sample = df_obs.groupby('batch', group_keys=False).apply(sample_or_take_all,n)
df_obs = df_obs.loc[sample.index.values]
df_expr = df_expr.loc[sample.index.values]

cnv_analysis(df_expr,df_gene,df_obs,tag)


### get expr from recons data 
tag='recons'
adata_recons = ad.read_h5ad('data/figure3_unique_recons.h5ad')
df_expr = adata_recons.to_df()
df_expr = df_expr.loc[:,present_genes]
df_expr = df_expr.loc[sample.index.values]
cnv_analysis(df_expr,df_gene,df_obs,tag)