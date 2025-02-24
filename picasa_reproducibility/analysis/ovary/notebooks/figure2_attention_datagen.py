import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil,util
from scipy.sparse import csr_matrix
import torch
import numpy as np


sample ='ovary'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'

############ read original data as adata list

ddir = pp+sample+'/model_data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=25:
		break

picasa_data = batch_map

########## add leiden label 

dfleiden = pd.read_csv('data/figure1_umap_coordinates.csv.gz',index_col=0)

dfg = dfleiden.groupby(['c_leiden','celltype']).count()['c_umap1'].reset_index()
celltype_sum = dict(dfg.groupby('c_leiden')['c_umap1'].sum())
dfg['ncount'] = [x/celltype_sum[y] for x,y in zip(dfg['c_umap1'],dfg['c_leiden'])]
dfg.sort_values(['c_leiden','ncount'],ascending=False,inplace=True)

dfg[ (dfg.celltype=='Endothelial') & (dfg.c_umap1==70) ]

dfg.drop_duplicates(subset='c_leiden',inplace=True)

dfg = dfg.reset_index(drop=True).copy()


## fix issue of very low endothelial cells
dfg.iloc[38,1] = 'Endothelial'
dfg.iloc[38,2] = 70

dfg['p_label'] = ['Common'+str(x)+'/'+y for x,y in zip (dfg['c_leiden'],dfg['celltype'])]

dfg.sort_values(['c_umap1','celltype'],ascending=False,inplace=True)
dfg.drop_duplicates(subset='celltype',inplace=True)


dfleiden['p_label'] = ['Common'+str(x)+'/'+y for x,y in zip (dfleiden['c_leiden'],dfleiden['celltype'])]


dfleiden = dfleiden[dfleiden['p_label'].isin(dfg['p_label'])]
dfleiden['p_label'].value_counts()

n=100
dfleiden = dfleiden.groupby(['batch', 'p_label'], group_keys=False).apply(lambda x: x.sample(min(len(x), n)))

p_label_dict = {x.split('@')[0]:y for x,y in zip(dfleiden.index.values,dfleiden['p_label'])}

#### now apply label to all data

picasa_data_updated = {}
for d in picasa_data:
	c_cells = dfleiden[dfleiden['batch']==d].index.values
	c_cells = [x.split('@')[0] for x in c_cells]
	
	df_x = picasa_data[d].to_df()
	df_x = df_x.loc[c_cells]
	obs_new = picasa_data[d].obs.loc[c_cells].copy()  
	var_new = picasa_data[d].var.copy()  

	c_adata = ad.AnnData(X=csr_matrix(df_x.values), obs=obs_new, var=var_new)
	c_adata.obs['p_label'] = [p_label_dict[x] for x in c_adata.obs.index.values]
	picasa_data_updated[d] = c_adata
	


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')


nn_params = picasa_adata.uns['nn_params']
nn_params['device'] = 'cpu'
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/model_results/picasa_common.model', map_location=torch.device(nn_params['device'])))


patient_analyzed = []


df = pd.DataFrame()

sel_patients = ['EOC3','EOC443','EOC136','EOC733','EOC372']

for pairs in picasa_adata.uns['adata_pairs']:
	
	p1 = picasa_adata.uns['adata_keys'][pairs[0]]
	p2 = picasa_adata.uns['adata_keys'][pairs[1]]

	if p1 not in patient_analyzed and p1 in sel_patients:
		adata_p1 = picasa_data_updated[p1]
		adata_p2 = picasa_data_updated[p2]

		nbr_map ={}
		nbr_map[p1+'_'+p2] = util.generate_neighbours(adata_p2,adata_p1,p1+p2)
 
		data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map[p1+'_'+p2],'cpu',batch_size=10)
  
		eval_total_size=adata_p1.shape[0]
		main_attn,main_y = model.eval_attention_common(picasa_common_model,data_loader,eval_total_size)

		##############################################


		unique_celltypes = adata_p1.obs['p_label'].unique()
		num_celltypes = len(unique_celltypes)


		for idx, ct in enumerate(unique_celltypes):
			
			ct_ylabel = adata_p1.obs[adata_p1.obs['p_label'] == ct].index.values
			ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]

			min_cells = 1
			if len(ct_yindxs) < min_cells:
				continue

			df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
								index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
			np.fill_diagonal(df_attn.values, 0)
			df_attn.index = [p1+'@'+ct+'_'+x for x in df_attn.index]
			df = pd.concat([df, df_attn], axis=0)
			print(p1,ct,len(ct_yindxs),df.shape)
			patient_analyzed.append(p1)
	
df.to_csv('data/figure2_attention_scores.csv.gz',compression='gzip')
