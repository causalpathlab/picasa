import anndata as ad
import pandas as pd
import os 
import glob 
from picasa import model,dutil
import torch
import numpy as np


sample ='brca'
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


############ read model results as adata 
wdir = pp+sample
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')


nn_params = picasa_adata.uns['nn_params']
nn_params['device'] = 'cpu'
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/model_results/picasa_common.model', map_location=torch.device(nn_params['device'])))


patient_analyzed = []


df = pd.DataFrame()

## select 3 from both TNBC,ER,HER groups 
sel_patients = [
    'CID4495','CID44971','CID44991',
    'CID4471','CID4290A','CID4535',
    'CID3586','CID4066','CID3921'
    ]


for pairs in picasa_adata.uns['adata_pairs']:
    
	p1 = picasa_adata.uns['adata_keys'][pairs[0]]
	p2 = picasa_adata.uns['adata_keys'][pairs[1]]

	if p1 not in patient_analyzed and p1 in sel_patients:
		adata_p1 = picasa_data[p1]
		adata_p2 = picasa_data[p2]
		df_nbr = picasa_adata.uns['nbr_map']
		df_nbr = df_nbr[df_nbr['batch_pair']==p1+'_'+p2]
		nbr_map = {x:(y,z) for x,y,z in zip(df_nbr['key'],df_nbr['neighbor'],df_nbr['score'])}

		data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map,'cpu',batch_size=10)
		eval_total_size=1000
		main_attn,main_y = model.eval_attention_common(picasa_common_model,data_loader,eval_total_size)

		##############################################


		unique_celltypes = adata_p1.obs['celltype'].unique()
		num_celltypes = len(unique_celltypes)


		for idx, ct in enumerate(unique_celltypes):
			
			ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
			ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]

			min_cells = 25
			if len(ct_yindxs) < min_cells:
				continue

			df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
								index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
			np.fill_diagonal(df_attn.values, 0)
			df_attn.index = [p1+'@'+ct+'_'+x for x in df_attn.index]
			df = pd.concat([df, df_attn], axis=0)
			print(p1,ct,len(ct_yindxs),df.shape)
			patient_analyzed.append(p1)
	
df.to_csv(wdir+'/notebooks/data/figure2_attention_scores.csv.gz',compression='gzip')
