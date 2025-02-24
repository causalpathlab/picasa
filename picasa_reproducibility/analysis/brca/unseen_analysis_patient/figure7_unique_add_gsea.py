import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


sample ='brca'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_brca.h5ad')



num_batches = len(picasa_adata.obs['batch'].unique())
input_dim = adata.shape[1]
nn_params = picasa_adata.uns['nn_params']
enc_layers = [128,25]
unique_latent_dim = nn_params['latent_dim']
common_latent_dim = nn_params['latent_dim']
dec_layers = [128,128]
nn_params['device'] = 'cpu'


picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(nn_params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+'/model_results/picasa_unique.model', map_location=torch.device(nn_params['device'])))

picasa_unique_model.eval()


df_w = pd.DataFrame(picasa_unique_model.zinb_scale.weight.data.detach().cpu().numpy())

from scipy.stats import zscore

df_w = zscore(df_w, axis=0) 
df_w = df_w.T
df_w.columns = adata.var_names


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

selected_topics = ['u5', 'u15','u22','u35', 'u48','u55','u96']

df_w.reset_index(inplace=True)
df_w['index'] = ['u'+str(x) for x in df_w['index']]
df_w = df_w.loc[df_w['index'].isin(selected_topics)]

df_w.set_index('index',inplace=True)


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
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from plotnine import *

available_libraries = gp.get_library_name(organism="Human")

dbs = [
'KEGG_2021_Human',
'MSigDB_Hallmark_2020',
# 'Reactome_2022',
'Reactome_Pathways_2024'
]



unique_celltypes = list(ranked_gene_list.keys())

pval_col = 'FDR q-val' 
nes_col = 'NES'  
pval_col_log = '-log10(p-val)'
min_size, max_size = 10, 500  
top_n_pathways = 5

all_pathway_results = {}

df_main = pd.DataFrame()

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

			df_gsea["ct"] = factor
			df_result = pd.concat([df_result, df_gsea], axis=0)

		df_result[pval_col] = pd.to_numeric(df_result[pval_col], errors='coerce')
		df_result[nes_col] = pd.to_numeric(df_result[nes_col], errors='coerce')
		df_result[pval_col_log] = -np.log10(df_result[pval_col]+1e-8)
		df_result[pval_col_log] = df_result[pval_col_log].clip(lower=0, upper=4)
		df_result[nes_col] = df_result[nes_col].clip(lower=-2, upper=2)
  
		# df_result.reset_index(inplace=True)
		# pivot_df = df_result.pivot(index="Term", columns="ct", values=nes_col)
		# row_linkage = linkage(pivot_df, method="ward")
		# col_linkage = linkage(pivot_df.T, method="ward")
		# row_order = leaves_list(row_linkage)
		# col_order = leaves_list(col_linkage)
  
		# df_result["Term"] = pd.Categorical(df_result["Term"], categories=pivot_df.index[row_order], ordered=True)
		# df_result["ct"] = pd.Categorical(df_result["ct"], categories=pivot_df.columns[col_order], ordered=True)


		df_result['Term'] = [x.replace('Cells','') for x in df_result.index.values]
		df_result['Term'] = [x[:50] for x in df_result['Term']]
		df_result['Term'] = [db.split('_')[0]+'/'+x for x in df_result['Term']]

		df_main = pd.concat([df_main,df_result],axis=0)

	# 	plt.figure(figsize=(25, 10))
	# 	p = (ggplot(df_result, aes(x='ct', y='Term', color='NES', size=pval_col_log)) 
	# 			# + geom_point()
	# 			+ geom_point(shape='o')
	# 			+ scale_color_gradient(low="skyblue", high="green")				+ scale_size_continuous(range=(0, 4))
	# 			+ theme(panel_grid=element_blank(),  
	# 					panel_background=element_blank(),
	# 					axis_line=element_blank(),  
	# 					axis_ticks=element_blank(),  
	# 					axis_text_x=element_text(rotation=45, hjust=1),
	# 					plot_background=element_rect(fill='white', color='white',
	# 					)  
	# 					)  
	# 	)

	# 	plt.title(f'{pval_col} Score')
	# 	plt.xticks(rotation=90)
	# 	plt.tight_layout()
	# 	p.save(f'results/figure2_attention_gsea_{db}.pdf')
	# 	plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print(f'Failed.....{db}')


df_main.to_csv('results/figure7_unique_patient_gsea.csv')

pivot_df = df_main.pivot(index="Term", columns="ct", values=nes_col)
def unique_max(row):
    max_value = row.max()
    return (row >= max_value).sum() == 1  

pivot_df = pivot_df[pivot_df.apply(unique_max, axis=1)]


row_linkage = linkage(pivot_df, method="ward")
col_linkage = linkage(pivot_df.T, method="ward")
row_order = leaves_list(row_linkage)
col_order = leaves_list(col_linkage)

df_main = df_main[df_main['Term'].isin(pivot_df.index.values)]

df_main["Term"] = pd.Categorical(df_main["Term"], categories=pivot_df.index[row_order], ordered=True)
df_main["ct"] = pd.Categorical(df_main["ct"], categories=pivot_df.columns[col_order], ordered=True)


p = (ggplot(df_main, aes(x='ct', y='Term', color='NES', size=pval_col_log)) 
		# + geom_point()
		+ geom_point(shape='o')
		+ scale_color_gradient(low="skyblue", high="green")				+ scale_size_continuous(range=(0, 4))
		+ theme(panel_grid=element_blank(),  
				panel_background=element_blank(),
				axis_line=element_blank(),  
				axis_ticks=element_blank(),  
				axis_text_x=element_text(rotation=45, hjust=1),
				plot_background=element_rect(fill='white', color='white',
				)  
				)  
)

plt.title(f'{pval_col} Score')
plt.xticks(rotation=90)
# plt.tight_layout()
p.save(f'results/figure7_unique_patient_gsea.pdf',height=10,width=8,limitsize=False)
plt.close()


###### get numbers for paper

import pandas as pd
df = pd.read_csv('results/figure7_unique_patient_gsea.csv')
col = 'NOM p-val'
df.sort_values(col,ascending=True,inplace=True)

for ct in df.ct.unique():
    print('#############################')
    print(ct)
    print(df[df['ct']==ct][['Term.1',col,'NES']])
    
    