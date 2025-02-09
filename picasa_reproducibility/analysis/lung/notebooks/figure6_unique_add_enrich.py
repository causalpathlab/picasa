import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from plotnine import * 


df_w = pd.read_csv('data/figure6_unique_add_topic_gene.csv.gz',index_col=0)

####################################


def generate_gene_ranking(df,n_gene):

	gene_ranking = {}
	for x in range(df.shape[0]):
		h_gtab = df.T.iloc[:,x].sort_values(ascending=False)[:n_gene].reset_index()
		h_gtab.columns = ['gene','val']
		h_genes = h_gtab['gene'].values
  
		l_gtab = df.T.iloc[:,x].sort_values(ascending=True)[:n_gene].reset_index()
		l_gtab.columns = ['gene','val']
		l_genes = l_gtab['gene'].values
		l_genes = l_genes[::-1]

		genes = np.concatenate([h_genes,l_genes])
  
		gene_ranking['k'+str(x)] = genes

	return gene_ranking


####################################

df_w = pd.read_csv('data/figure6_unique_add_topic_gene.csv.gz',index_col=0)

#####################################

print(df_w.shape)

n_gene = 50
gene_ranking = generate_gene_ranking(df_w,n_gene)
    
all_deg_clusters = gene_ranking

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
		plt.savefig('results/figure6_unique_add_enrich_'+db+'.png')
		plt.close()
	
	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)