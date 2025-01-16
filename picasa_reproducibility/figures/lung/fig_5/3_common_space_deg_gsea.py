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
    

wdir = wdir + '/fig_2/'
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

sc.tl.rank_genes_groups(adata, groupby='cluster', method='wilcoxon')
deg_results = adata.uns['rank_genes_groups'] 

all_deg_dfs = []
for group in adata.obs['cluster'].unique(): 
    deg_df = sc.get.rank_genes_groups_df(adata, group=group)
    deg_df['cluster'] = group  
    all_deg_dfs.append(deg_df)
all_deg_df = pd.concat(all_deg_dfs, ignore_index=True)

deg_filtered = all_deg_df[all_deg_df['pvals_adj'] < 0.01]


unique_celltypes = adata.obs['cluster'].unique()
num_celltypes = len(unique_celltypes)


ranked_gene_list = {}
for idx, ct in enumerate(unique_celltypes):
    
    cluster_degs = deg_filtered[deg_filtered['cluster'] == ct]
    ranked_genes = cluster_degs[['names', 'logfoldchanges']].sort_values(by='logfoldchanges', ascending=False)
    
    ranked_genes.columns = ['Gene','Score']
    ranked_gene_list[ct] = ranked_genes

import gseapy as gp

available_libraries = gp.get_library_name(organism="Human")

dbs = [
# 'Azimuth_2023',
# 'Azimuth_Cell_Types_2021',
'BioPlanet_2019',
# 'CellMarker_Augmented_2021',
'GO_Biological_Process_2023',
'GO_Cellular_Component_2023',
'GO_Molecular_Function_2023',
'GTEx_Tissues_V8_2023',
'GWAS_Catalog_2023',
'KEGG_2021_Human',
'MSigDB_Hallmark_2020',
# 'PanglaoDB_Augmented_2021',
'Reactome_2022',
'WikiPathways_2024_Human'
 
]

score_col = 'NES'
for db in dbs:
	try:
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")


		top_n_pathways = 3
		top_pathways = []
		for factor in unique_celltypes:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=10,  
								max_size=100,  
								permutation_num=1000,
								outdir=None)  
			
			top_pathways.append(gsea_res.res2d.sort_values(by=score_col, ascending=False).head(top_n_pathways)['Term'].values)


		tps = []
		for tp in np.concatenate(top_pathways): 
			if tp not in tps: 
				tps.append(tp)
		top_pathways = np.array(tps)



		results = {}
		results['pathways'] = top_pathways

		for factor in unique_celltypes:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=15,  
								max_size=100,  
								permutation_num=1000,
								outdir=None)  
			
			df_gsea = pd.DataFrame(gsea_res.res2d)
			
			no_pathways =  np.setdiff1d(df_gsea['Term'].values, results['pathways']).tolist() + np.setdiff1d(results['pathways'], df_gsea['Term'].values).tolist()
		
			df_gsea.set_index('Term',inplace=True)

			df_gsea = df_gsea[[score_col]]
		
			for pathway in no_pathways:
				df_gsea.loc[pathway] = 0.0
		
			results[factor] = df_gsea.loc[results['pathways']][score_col].values

		df_result = pd.DataFrame(results)
		df_result.set_index('pathways',inplace=True)
		df_result = df_result.astype(float)


		from matplotlib.colors import LinearSegmentedColormap
		colors = ['darkblue', 'lightblue', 'white', 'lightcoral', 'darkred']
		plt.rcParams.update({'font.size': 10})
		plt.figure(figsize=(35, 15))
		custom_cmap = LinearSegmentedColormap.from_list('custom_vlag', colors)

		if (df_result < 0).any().any():
			col_p = sns.color_palette("viridis", as_cmap=True)
		else:
			col_p= sns.color_palette("Blues", as_cmap=True)
		
		sns.clustermap(df_result, 
			yticklabels=df_result.index,  
			xticklabels=df_result.columns,
			annot=False, cmap=col_p),# cbar_kws={'label': score_col+' Score'})
		plt.title(score_col+" score")
		plt.xticks(rotation=90)
		plt.savefig(wdir+'/results/gsea_'+db+'.png')
		plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)