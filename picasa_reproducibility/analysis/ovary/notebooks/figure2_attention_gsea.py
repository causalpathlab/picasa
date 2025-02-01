import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('data/figure2_attention_scores.csv.gz',index_col=0)

##############################################

dfl = pd.DataFrame(df.index.values)
dfl['patient'] = [x.split('@')[0] for x in dfl[0]]
dfl['celltype'] = [x.split('@')[1].split('_')[0] for x in dfl[0]]
dfl['gene'] = [x.split('_')[1] for x in dfl[0]]
dfl.drop(0,axis=1,inplace=True)
dfl.reset_index(inplace=True)

##################

unique_celltypes = dfl['celltype'].unique()
num_celltypes = len(unique_celltypes)


ranked_gene_list = {}
top_n = 1000
for idx, ct in enumerate(unique_celltypes):
	ct_ylabel = dfl[dfl['celltype'] == ct].index.values
	df_attn = df.iloc[ct_ylabel,:].copy()
	
	df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
	df_attn = df_attn.groupby('gene').mean()
 
	df_attn = df_attn.unstack().reset_index()

	df_attn.columns = ['Gene1','Gene2','Score']
	df_attn = df_attn.sort_values('Score',ascending=False)
 
	df_attn = df_attn.iloc[:top_n,:]

	df_attn = pd.melt(df_attn, 
                    id_vars=['Score'], 
                    value_vars=['Gene1', 'Gene2'], 
                    var_name='Gene_Type', 
                    value_name='Gene')

	df_attn = df_attn[['Gene','Score']]
	df_attn = df_attn.drop_duplicates(subset='Gene') 
	df_attn['Score'] = df_attn['Score'] + (0.1*(df_attn.index.values[::-1]))
	df_attn['Gene'] = df_attn['Gene'].str.upper()
	ranked_gene_list[ct] = df_attn.reset_index(drop=True)
	

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
# 'KEGG_2021_Human',
# 'MSigDB_Hallmark_2020',
'PanglaoDB_Augmented_2021',
# 'Reactome_2022',
# 'WikiPathways_2024_Human'
 
]

score_col = 'NES'
for db in dbs:
	try:
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")


		top_n_pathways = 5
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
		plt.savefig('results/figure2_attention_gsea_'+db+'.png')
		plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)