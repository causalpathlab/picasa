import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


sample ='ovary'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

adata = ad.read_h5ad(wdir+'/model_data/all_ovary.h5ad')



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


selected_topics = ['k15', 'k40', 'k47', 'k49', 'k88', 'k99']

df_w.reset_index(inplace=True)
df_w['index'] = ['k'+str(x) for x in df_w['index']]
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


score_col = 'FDR q-val' 
nes_col = 'NES'  
pval_th = 0.01
df_main = pd.DataFrame()

unique_celltypes = list(ranked_gene_list.keys())

for db in dbs:
	try:
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")

		top_n_pathways = 10
		top_pathways = []
		for factor in unique_celltypes:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=15,  
								max_size=100,  
								permutation_num=1000,
								outdir=None)  

			# gsea_res.res2d = gsea_res.res2d[gsea_res.res2d['FDR q-val']<pval_th]
			top_pathways.append(gsea_res.res2d.sort_values(by=nes_col, ascending=False).head(top_n_pathways)['Term'].values)

		tps = []
		for tp in np.concatenate(top_pathways): 
			if tp not in tps: 
				tps.append(tp)
		top_pathways = np.array(tps)

		df_result = pd.DataFrame()

		for factor in unique_celltypes:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=3,  
								max_size=1000,  
								permutation_num=1000,
								outdir=None)  
			
			df_gsea = pd.DataFrame(gsea_res.res2d)
			
			no_pathways =  np.setdiff1d(top_pathways, df_gsea['Term'].values).tolist()
		
			df_gsea.set_index('Term', inplace=True)

			df_gsea = df_gsea[[score_col, nes_col]]

			for pathway in no_pathways:
				df_gsea.loc[pathway] = 0.0

			df_gsea = 	df_gsea.loc[top_pathways]
			df_gsea['ct'] = factor
			df_result = pd.concat([df_result,df_gsea],axis=0)


		df_result[score_col] = pd.to_numeric(df_result[score_col], errors='coerce')
		df_result[nes_col] = pd.to_numeric(df_result[nes_col], errors='coerce')
		df_result[score_col] = -np.log10(df_result[score_col]+1e-8)
		df_result[score_col] = df_result[score_col].clip(lower=0, upper=4)
		df_result[nes_col] = df_result[nes_col].clip(lower=-2, upper=2)  
  
		df_result.reset_index(inplace=True)
		df_result.rename(columns={'index':'Term'},inplace=True)

		pivot_df = df_result.pivot(index="Term", columns="ct", values="FDR q-val")
		row_linkage = linkage(pivot_df, method="ward")
		col_linkage = linkage(pivot_df.T, method="ward")
		row_order = leaves_list(row_linkage)
		col_order = leaves_list(col_linkage)

		df_result["Term"] = pd.Categorical(df_result["Term"], categories=pivot_df.index[row_order], ordered=True)
		df_result["ct"] = pd.Categorical(df_result["ct"], categories=pivot_df.columns[col_order], ordered=True)


		df_result['Term'] = [x.replace('Cells','') for x in df_result['Term']]

		p = (ggplot(df_result, aes(x='ct', y='Term', color='NES', size='FDR q-val')) 
				# + geom_point()
				+ geom_point(shape='o')
				+ scale_color_gradient(low="skyblue", high="green")
				+ scale_size_continuous(range=(0, 4))
				+ theme(panel_grid=element_blank(),  
						panel_background=element_blank(),
						axis_line=element_blank(),  
						axis_ticks=element_blank(),  
						axis_text_x=element_text(rotation=45, hjust=1),
						plot_background=element_rect(fill='white', color='white')  
						)  
		)

		p.save(f'results/figure7_unique_add_gsea_'+db+'.pdf')
		plt.tight_layout()
		plt.title(f'{score_col} Score')
		plt.xticks(rotation=90)
		plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print(f'Failed.....{db}')


