import sys
import scanpy as sc
import matplotlib.pylab as plt
import seaborn as sns
import anndata as ad
import os 
import glob 
import numpy as np
import pandas as pd

sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
# sample = sys.argv[1] 
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'+sample


############ read original data as adata list


ddir = wdir+'/data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

picasa_data = batch_map


############ read model results as adata 

df_main = pd.DataFrame()
for p_ad in picasa_data:
	df_main = pd.concat([df_main,picasa_data[p_ad].to_df()],axis=0)
	

wdir = wdir + '/fig_5/'
df = pd.read_csv(wdir+'/results/common_space_selected_cells.csv.gz',compression='gzip',index_col=0)
df.index = ['@'.join(x.split('@')[:2])for x in df.index.values]

df_main = df_main.loc[df.index.values]


adata = ad.AnnData(
	X=df_main.values,  
	obs=pd.DataFrame(index=df_main.index), 
	var=pd.DataFrame(index=df_main.columns)  
)

adata.obs['cluster'] = df['cluster'].values


import scanpy as sc
'''
rank based on gene expression not DEG 

renormalize -- genes are normalized across all cells

save data before plotting

'''

###### use normalized expression
# all_deg_clusters = {}
# top_n = 100
# for clust in adata.obs['cluster'].unique():
# 	all_deg_clusters[clust] = adata[adata.obs['cluster']==clust].to_df().mean().sort_values(ascending=False).index.values[:top_n]

    
###### use scanpy DEG
sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon')
deg_results = adata.uns['rank_genes_groups'] 
all_deg_clusters = {}
p_th = 0.05
lfc_th = 1
top_n = 25
for group in adata.obs['cluster'].unique(): 
	deg_df = sc.get.rank_genes_groups_df(adata, group=group)
	deg_df = deg_df[(deg_df['pvals_adj'] < p_th) & (deg_df['logfoldchanges'] > lfc_th)]
	deg_df = deg_df.iloc[:top_n,:]
	all_deg_clusters[group] = deg_df['names'].values



import gseapy as gp



available_libraries = gp.get_library_name(organism="Human")

dbs = [
'Azimuth_2023',
'Azimuth_Cell_Types_2021',
'BioPlanet_2019',
'CellMarker_Augmented_2021',
'GO_Biological_Process_2023',
'GO_Cellular_Component_2023',
'GO_Molecular_Function_2023',
'GTEx_Tissues_V8_2023',
'GWAS_Catalog_2023',
'KEGG_2021_Human',
'MSigDB_Hallmark_2020',
'PanglaoDB_Augmented_2021',
'Reactome_2022',
'WikiPathways_2024_Human'
 
]

p_th = 0.01
for db in dbs:
	try:
		result = []
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")

		for group in all_deg_clusters.keys():
			
			sig_genes = all_deg_clusters[group].tolist()
			
			enr = gp.enrichr(
				gene_list=sig_genes,  
				gene_sets=db           
			)

			enr.res2d = enr.res2d.loc[enr.res2d['Adjusted P-value']<p_th,:]

			for row in enr.res2d.iterrows():
				result.append([db,group,row[1]['Term'],row[1]['Adjusted P-value']])




		df_res = pd.DataFrame(result,columns =['db','group','p_names','p_scores'])

		# df_res.to_csv(wdir+'/results/gene_enrichment.csv.gz',compression='gzip')
		# df_res= pd.read_csv(wdir+'/results/gene_enrichment.csv.gz',index_col=0)

		df_res['p_scores'] = -np.log10(df_res['p_scores'])
		df_res['p_names'] = [' '.join(x.split(' ')[:5])for x in df_res['p_names']]
		df_res['p_names'] = [str(x)+'('+y+')'for x,y in zip(df_res['p_names'],df_res['db'])]

		pivot_df = df_res.pivot_table(index='group', columns='p_names', values='p_scores', fill_value=0)

		pivot_df[pivot_df>5] = 5
		pivot_df = pivot_df.T
		sns.clustermap(pivot_df,
					yticklabels=pivot_df.index,  
					xticklabels=pivot_df.columns,
					cmap=sns.color_palette("rocket", as_cmap=True),cbar_kws={"label": "-log10(adjPval) Score"})
		plt.title(" score")
		plt.savefig(wdir+'/results/enrichment_cmap'+db+'.png')
		plt.close()
	
	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)
