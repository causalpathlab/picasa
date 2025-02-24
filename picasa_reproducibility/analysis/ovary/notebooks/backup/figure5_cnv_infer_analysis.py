import scanpy as sc
import infercnvpy as cnv
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd 
from plotnine import * 
import sys


sample ='ovary'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'
wdir = pp + sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')
df_obs = picasa_adata.obs.copy()
df_obs.index = [x.split('@')[0] for x in df_obs.index.values]

df_gene = pd.read_csv(wdir+'/model_data/df_gene_coord.csv.gz',index_col=0)



def sample_or_take_all(group, n):
	return group.sample(n=min(len(group), n), random_state=42)

def cnv_analysis(df_expr,df_gene,df_obs,tag):

	present_genes = [ x for x in df_expr.columns if x  in df_gene.index.values]


	df_gene = df_gene.loc[present_genes]
	df_expr = df_expr.loc[:,present_genes]
	df_obs = df_obs.loc[df_expr.index.values]
	
	n=2500
	sample = df_obs.groupby('celltype', group_keys=False).apply(sample_or_take_all,n)

	df_obs = df_obs.loc[sample.index.values]
	df_expr = df_expr.loc[sample.index.values]

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
	)



	plt.xlabel("chr", fontsize=14)
	plt.ylabel("patient", fontsize=14)
	plt.rcParams.update({'font.size': 14})	
	cnv.pl.chromosome_heatmap(adata, 
		figsize=(25, 20),
		groupby="batch")
		
	plt.savefig('results/figure5_cnv_'+tag+'.pdf')


	cnv.tl.pca(adata)
	cnv.pp.neighbors(adata,n_neighbors=30)
	cnv.tl.leiden(adata,resolution=0.5)
	cnv.tl.umap(adata,min_dist=1)
	cnv.pl.umap(adata, color=['cnv_leiden','celltype','batch'])

	plt.savefig('results/figure5_cnv_umap_'+tag+'.png')

	# df_cnv = pd.DataFrame(adata.obsm['X_cnv'].todense())
	# df_cnv.to_csv('results/cnv_'+tag+'.csv.gz',compression='gzip')

	# plot_prop(adata.obs.copy(),tag)
 
tag = sys.argv[1]

if tag=='orig':
	df_expr = pd.DataFrame()
	for p1 in picasa_adata.obs['batch'].unique():	
		df_expr_p = pd.read_csv('data/figure5_cnv_x_orig_'+p1+'.csv.gz',index_col=0)
		df_expr = pd.concat([df_expr,df_expr_p])
	cnv_analysis(df_expr,df_gene,df_obs,tag)

elif tag=='recons':
	df_expr = pd.DataFrame()
	for p1 in picasa_adata.obs['batch'].unique():	
		df_expr_p = pd.read_csv('data/figure5_cnv_x_recons_'+p1+'.csv.gz',index_col=0)
		df_expr = pd.concat([df_expr,df_expr_p])
 
	cnv_analysis(df_expr,df_gene,df_obs,tag)