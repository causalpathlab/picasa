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

######## fix cell type
cmap = {
'EOC':'Malignant',
'Macrophages':'Monocyte', 
'Plasma':'Plasma',
'CAF':'Fibroblasts', 
'Endothelial':'Endothelial',
'T':'T', 
'B':'B',
'DC':'DC',
'NK':'NK', 
'Mast':'Mast'
}

dfl['celltype'] = [cmap[x] for x in dfl['celltype']]

dfl = dfl[dfl['celltype']!='Epithelial']

##################

unique_celltypes = dfl['celltype'].unique()
num_celltypes = len(unique_celltypes)


ranked_gene_list = {}
top_n = 2000 ## this is ok as total gene is 2k so all ranked
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
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from plotnine import *

available_libraries = gp.get_library_name(organism="Human")

dbs = [
	'PanglaoDB_Augmented_2021' 
	]

pval_col = 'FDR q-val' 
nes_col = 'NES'  
min_size, max_size = 10, 500  
top_n_pathways = 5

all_pathway_results = {}

for db in dbs:
	try:
		print(f"Running GSEA for {db}")
		gene_set_library = gp.get_library(name=db, organism="Human")

		all_pathways = []

		for factor in unique_celltypes:
			gsea_res = gp.prerank(
				rnk=ranked_gene_list[factor],
				gene_sets=gene_set_library,
				min_size=min_size,
				max_size=max_size,
				permutation_num=1000,
				outdir=None,
			)

			all_pathway_results[factor] = gsea_res.res2d

			top_paths = (
				gsea_res.res2d.sort_values(by=nes_col, ascending=False)
				.head(top_n_pathways)["Term"]
				.values
			)
			all_pathways.extend(top_paths)

		selected_pathways = np.unique(all_pathways)

		df_result = pd.DataFrame()

		for factor in unique_celltypes:
			df_gsea = all_pathway_results[factor].set_index("Term")

			df_gsea = df_gsea.reindex(selected_pathways)
			## 1 for pval	
			df_gsea[pval_col] = df_gsea[pval_col].fillna(1.0)
			## 0 for nes
			df_gsea[nes_col] = df_gsea[nes_col].fillna(0.0)  

			df_gsea = df_gsea[[pval_col, nes_col]]

			df_gsea["ct"] = factor
			df_result = pd.concat([df_result, df_gsea], axis=0)

		df_result[pval_col] = pd.to_numeric(df_result[pval_col], errors='coerce')
		df_result[nes_col] = pd.to_numeric(df_result[nes_col], errors='coerce')
		df_result[pval_col] = -np.log10(df_result[pval_col]+1e-8)
		df_result[pval_col] = df_result[pval_col].clip(lower=0, upper=4)
		df_result[nes_col] = df_result[nes_col].clip(lower=-2, upper=2)
		df_result.reset_index(inplace=True)
		
		pivot_df = df_result.pivot(index="Term", columns="ct", values="FDR q-val")
		row_linkage = linkage(pivot_df, method="ward")
		col_linkage = linkage(pivot_df.T, method="ward")
		row_order = leaves_list(row_linkage)
		col_order = leaves_list(col_linkage)
  
		df_result["Term"] = pd.Categorical(df_result["Term"], categories=pivot_df.index[row_order], ordered=True)
		df_result["ct"] = pd.Categorical(df_result["ct"], categories=pivot_df.columns[col_order], ordered=True)


		df_result['Term'] = [x.replace('Cells','') for x in df_result['Term']]

		plt.figure(figsize=(15, 10))
		p = (ggplot(df_result, aes(x='ct', y='Term', color='NES', size='FDR q-val')) 
				# + geom_point()
				+ geom_point(shape='s')
				+ scale_color_gradient(low="blue", high="red")
				+ scale_size_continuous(range=(0, 4))
				+ theme(panel_grid=element_blank(),  
						panel_background=element_blank(),
						axis_line=element_blank(),  
						axis_ticks=element_blank(),  
						axis_text_x=element_text(rotation=45, hjust=1),
						plot_background=element_rect(fill='white', color='white',
						)  
						)  
		)

		p.save(f'results/figure2_attention_gsea_{db}.pdf')
		plt.title(f'{pval_col} Score')
		plt.xticks(rotation=90)
		plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print(f'Failed.....{db}')

