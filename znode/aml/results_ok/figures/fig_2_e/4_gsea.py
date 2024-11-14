import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os


sample = 'aml'
wdir = 'znode/aml/'
cdir='results/figures/fig_2_e/'

directory = wdir+'/data'
pattern = 'aml_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('aml_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=10:
		break


file_name = file_names[0].replace('.h5ad','').replace('aml_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	'unq',
	wdir)

params = params = {'device': 'cuda', 
                'batch_size': 100, 
                'input_dim': 2000, 
                'embedding_dim': 3000, 
                'attention_dim': 15, 
                'latent_dim': 15, 
                'encoder_layers': [100,15], 
                'projection_layers': [15, 15], 
                'learning_rate': 0.001, 
                'lambda_loss': [1.0, 0.1, 0.0, 1.0], 
                'temperature_cl': 1.0, 
                'pair_search_method': 'approx_50', 
                'pair_importance_weight': 0.01, 
                'corruption_tol':5, 
                'cl_loss_mode': 'weighted', 
                'loss_clusters': 2, 
                'loss_threshold': 0.1, 
                'loss_weight': 0.5, 
                'epochs': 1, 
                'titration': 20
                }


import h5py as hf
from scipy.stats import zscore
import gseapy as gp

picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

# since we have only one pair 
p1,p2 = batch_keys[0],batch_keys[1]

adata_p1 = picasa_object.data.adata_list[p1]
adata_p2 = picasa_object.data.adata_list[p2]
nbr_map = {int(x):(int(y),float(z)) for x,y,z in list(picasa_h5[p1+'_'+p2])}

device = 'cpu'
picasa_object.set_nn_params(params)
picasa_object.nn_params['device'] = device
eval_batch_size = 10
eval_total_size = 1000
p1_attention,p1_ylabel = picasa_object.eval_attention(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)


picasa_h5.close()

	
df_umap = pd.read_csv(wdir+'results/df_umap.csv.gz')
# df_umap['cluster'] = ['c_'+str(x) for x in df_umap['cluster'].values] 	 
df_umap['cluster'] = [x for x in df_umap['cell_type'].values] 	 

# sel_clust =[
# 'c_6','c_7','c_11','c_14',	
#  'c_0','c_10','c_8','c_5',	
#  'c_2','c_1','c_3','c_9',	
#  'c_12','c_13','c_4'
# ]
# df_umap = df_umap.loc[df_umap['cluster'].isin(sel_clust)]

sel_clust = df_umap['cluster'].unique()


print(df_umap['cluster'].value_counts())


ranked_gene_list = {}
top_n = 500
for idx, ct in enumerate(sel_clust):
	ct_cells = df_umap[df_umap['cluster'] == ct]['cell'].values
	ct_yindxs = np.where(np.isin(p1_ylabel, ct_cells))[0]
	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
	df_attn = df_attn.unstack().reset_index()
	df_attn.fillna(0.0,inplace=True)
	df_attn.columns = ['Gene1','Gene2','Score']
	df_attn = df_attn.sort_values('Score',ascending=False)
	df_attn = df_attn[['Gene1','Score']]
	df_attn = df_attn.drop_duplicates(subset='Gene1') 
	df_attn = df_attn.iloc[:top_n,:].reset_index(drop=True)
	df_attn.columns = ['Gene','Score']
	df_attn['Score'] = df_attn['Score'] + (0.1*(df_attn.index.values[::-1]))
	df_attn['Gene'] = df_attn['Gene'].str.upper()
	ranked_gene_list[ct] = df_attn.reset_index(drop=True)
	


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

for db in dbs:
	try:
		print(db)
		gene_set_library = gp.get_library(name=db, organism="Human")


		top_n_pathways = 5
		top_pathways = []
		for factor in sel_clust:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=10,  
								max_size=100,  
								permutation_num=1000,
								outdir=None)  
			
			top_pathways.append(gsea_res.res2d.sort_values(by='NES', ascending=False).head(top_n_pathways)['Term'].values)


		tps = []
		for tp in np.concatenate(top_pathways): 
			if tp not in tps: 
				tps.append(tp)
		top_pathways = np.array(tps)



		results = {}
		results['pathways'] = top_pathways

		for factor in sel_clust:
			gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
								gene_sets=gene_set_library,
								min_size=15,  
								max_size=100,  
								permutation_num=1000,
								outdir=None)  
			
			df_gsea = pd.DataFrame(gsea_res.res2d)
			
			no_pathways =  np.setdiff1d(df_gsea['Term'].values, results['pathways']).tolist() + np.setdiff1d(results['pathways'], df_gsea['Term'].values).tolist()
		
			df_gsea.set_index('Term',inplace=True)

			df_gsea = df_gsea[['NES']]
		
			for pathway in no_pathways:
				df_gsea.loc[pathway] = 0.0
		
			results[factor] = df_gsea.loc[results['pathways']]['NES'].values

		df_result = pd.DataFrame(results)
		df_result.set_index('pathways',inplace=True)
		df_result = df_result.astype(float)


		from matplotlib.colors import LinearSegmentedColormap
		colors = ['darkblue', 'lightblue', 'white', 'lightcoral', 'darkred']
		plt.rcParams.update({'font.size': 10})
		plt.figure(figsize=(35, 15))
		custom_cmap = LinearSegmentedColormap.from_list('custom_vlag', colors)

		if (df_result < 0).any().any():
			col_p = sns.color_palette("vlag", as_cmap=True)
		else:
			col_p= sns.color_palette("Blues", as_cmap=True)
		
		sns.heatmap(df_result, annot=False, cmap=col_p, cbar_kws={'label': 'NES (Normalized Enrichment Score)'})
		plt.title("Heatmap of Enrichment Scores (NES) for Clusters and Pathways")
		plt.ylabel("Pathways")
		plt.xlabel("Clusters")
		plt.xticks(rotation=90)
		plt.savefig(wdir+cdir+'gsea_'+db+'.png')
		plt.close()

	except Exception as e:  
		print(f"An unexpected error occurred: {e}")
		print('Failed.....'+db)