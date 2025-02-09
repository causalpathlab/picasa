import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from plotnine import * 


df_w = pd.read_csv('data/figure6_unique_add_topic_gene.csv.gz',index_col=0)

df_w.set_index('index',inplace=True)

####################################

def generate_gene_ranking(df,n_gene):

	gene_ranking = {}
	for x in df.index.values:
		h_gtab = df.T.loc[:,x].sort_values(ascending=False)[:n_gene].reset_index()
		h_gtab.columns = ['gene','val']
		h_genes = h_gtab['gene'].values
  
		l_gtab = df.T.loc[:,x].sort_values(ascending=True)[:n_gene].reset_index()
		l_gtab.columns = ['gene','val']
		l_genes = l_gtab['gene'].values
		l_genes = l_genes[::-1]

		genes = np.concatenate([h_genes,l_genes])
  
		gene_ranking[x] = genes

	return gene_ranking


####################################

print(df_w.shape)

n_gene = 1000
gene_ranking = generate_gene_ranking(df_w,n_gene)

ranked_gene_list={}
for k,v in gene_ranking.items():
    df_c = df_w.loc[:,v]
    df_c = pd.DataFrame(df_c.loc[k].sort_values(ascending=False))
    df_c.reset_index(inplace=True)
    df_c.columns = ['Gene','Score']
    ranked_gene_list[k] = df_c
    

import gseapy as gp

available_libraries = gp.get_library_name(organism="Human")

dbs = [
# 'Azimuth_2023',
# 'Azimuth_Cell_Types_2021',
# 'BioPlanet_2019',
# 'CellMarker_Augmented_2021',
# 'GO_Biological_Process_2023',
# 'GO_Cellular_Component_2023',
# 'GO_Molecular_Function_2023',
# 'GTEx_Tissues_V8_2023',
# 'GWAS_Catalog_2023',
'KEGG_2021_Human',
# 'MSigDB_Hallmark_2020',
# 'MSigDB_Oncogenic_Signatures',
# 'PanglaoDB_Augmented_2021',
'Reactome_Pathways_2024',
'WikiPathways_2024_Human'
]

df_main = pd.DataFrame()

unique_celltypes = list(ranked_gene_list.keys())
score_col = 'NES'
for db in dbs:
	try:
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")


		top_n_pathways = 10
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
		plt.savefig('results/figure6_unique_add_gsea_'+db+'.png')
		plt.close()

		df_result.index = [x+'('+db.split('_')[0]+')' for x in df_result.index]
  
		df_main = pd.concat([df_main,df_result],axis=0)
  
	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)
  
th = 2
df_main[df_main>th] = th
df_main[df_main<-th] = -th
sns.clustermap(df_main, 
	yticklabels=df_main.index,  
	xticklabels=df_main.columns,
	annot=False, cmap=col_p,
 	figsize=(15, 25)),# cbar_kws={'label': score_col+' Score'})
plt.xticks(rotation=90)
plt.savefig('results/figure6_unique_add_gsea_all.png')
plt.close()
