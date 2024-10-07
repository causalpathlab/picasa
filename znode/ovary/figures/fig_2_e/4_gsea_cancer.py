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

import gseapy as gp

import glob
import os


sample = 'ovary'
wdir = 'znode/ovary/'
cdir = 'figures/fig_2_e/'


directory = wdir+'/data'
pattern = 'ovary_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break


file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('ovary_','')].X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,1.0],
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
		 'corruption_rate' : 0.0,
		'pair_importance_weight' : 0.01,
		'rare_ct_mode' : False, 
		  'num_clusters' : 5, 
		'rare_group_threshold' : 0.1, 
		'rare_group_weight': 2.0,
		'epochs': 1,
		'titration': 10
		}  

# picasa_object.estimate_neighbour(params['neighbour_method'])	


import h5py as hf
from scipy.stats import zscore


picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

# since we have only one pair 
p1,p2 = batch_keys[0],batch_keys[1]

adata_p1 = picasa_object.data.adata_list[p1]
adata_p2 = picasa_object.data.adata_list[p2]
nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

device = 'cpu'
picasa_object.set_nn_params(params)
picasa_object.nn_params['device'] = device
eval_batch_size = 10
eval_total_size = 1000
p1_attention,p1_ylabel = picasa_object.eval_attention(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)


picasa_h5.close()

	
df_umap = pd.read_csv(wdir+'results/df_umap_cancer.csv.gz')
df_umap['cluster'] = ['c_'+str(x) for x in df_umap['cluster'].values] 	 

sel_clust =[
 'c_0','c_4','c_2','c_3','c_1','c_5'
]

df_umap = df_umap.loc[df_umap['cluster'].isin(sel_clust)]

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
print(available_libraries)

dbs = [
# "MSigDB_Hallmark_2020",
# "MSigDB_Oncogenic_Signatures",
# "Reactome_2022",
# "GWAS_Catalog_2023",
"GO_Cellular_Component_2023",
"GO_Biological_Process_2023",
"CellMarker_2024",
]
for pathway_db in dbs:
	print(pathway_db)
	gene_set_library = gp.get_library(name=pathway_db, organism="Human")

	top_n_pathways = 5
	top_pathways = []
	for factor in sel_clust:
		gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
							gene_sets=gene_set_library,
							min_size=15,  
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
	plt.figure(figsize=(25, 15))
	custom_cmap = LinearSegmentedColormap.from_list('custom_vlag', colors)

	sns.heatmap(df_result, annot=False, cmap='vlag', cbar_kws={'label': 'NES (Normalized Enrichment Score)'})
	plt.title("Heatmap of Enrichment Scores (NES) for Clusters and Pathways")
	plt.ylabel("Pathways")
	plt.xlabel("Clusters")
	plt.xticks(rotation=90)
	plt.savefig(wdir+cdir+'gsea_cancer_'+pathway_db+'.png')
	plt.close()


