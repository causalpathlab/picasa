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

selected_topics = ['k22','k35','k55','k96','k104']

df_w.reset_index(inplace=True)
df_w['index'] = ['k'+str(x) for x in df_w['index']]
df_w = df_w.loc[df_w['index'].isin(selected_topics)]
df_w.drop(columns={'index'},inplace=True)
df_w.reset_index(drop=True,inplace=True)


print(df_w.shape)

n_gene = 2000
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
# 'BioCarta_2016',
# 'CellMarker_Augmented_2021',
# 'GO_Biological_Process_2023',
# 'GO_Cellular_Component_2023',
# 'GO_Molecular_Function_2023',
# 'GTEx_Tissues_V8_2023',
# 'GWAS_Catalog_2023',
# 'KEGG_2021_Human',
'MSigDB_Hallmark_2020',
# 'MSigDB_Oncogenic_Signatures',
# 'PanglaoDB_Augmented_2021',
# 'Reactome_Pathways_2024',
# 'WikiPathways_2024_Human'
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
		df_result.columns = selected_topics
		df_result = df_result.astype(float)

		df_result.index = [x+'('+db.split('_')[0]+')' for x in df_result.index]
  
		df_main = pd.concat([df_main,df_result],axis=0)
  
	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)
  
df_main = df_main[df_main.sum(1)>1]
max_thresh = -1 * df_main.min().min()
df_main[df_main>max_thresh] = max_thresh
df_main[df_main<-max_thresh] = -max_thresh
sns.clustermap(df_main, 
	yticklabels=df_main.index,  
	xticklabels=df_main.columns,
	annot=False,cmap='RdBu_r',
 	figsize=(15, 25)),# cbar_kws={'label': score_col+' Score'})
plt.xticks(rotation=90)

plt.savefig('results/figure7_unique_add_gsea_all.pdf')
plt.close()
