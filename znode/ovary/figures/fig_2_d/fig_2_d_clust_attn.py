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


sample = 'ovary'
wdir = 'znode/ovary/'
cdir = 'figures/fig_2_d/'


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

	
df_umap = pd.read_csv(wdir+'results/df_umap.csv.gz')
df_umap['cluster'] = ['c_'+str(x) for x in df_umap['cluster'].values] 	 

sel_clust =[
'c_6','c_7','c_11',	
 'c_0','c_3','c_1',
 'c_12','c_13','c_4'
]
df_umap = df_umap.loc[df_umap['cluster'].isin(sel_clust)]

print(df_umap['cluster'].value_counts())

unique_celltypes = df_umap['cluster'].unique()
num_celltypes = len(unique_celltypes)
top_genes = []
top_n = 10
for idx, ct in enumerate(sel_clust):
	ct_cells = df_umap[df_umap['cluster'] == ct]['cell'].values
	ct_yindxs = np.where(np.isin(p1_ylabel, ct_cells))[0]
	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
	df_attn = df_attn.unstack().reset_index()
	df_attn = df_attn.sort_values(0,ascending=False)
	top_genes.append(df_attn['level_0'].unique()[:top_n])

tgs = []
for tg in np.array(top_genes).flatten(): 
    if tg not in tgs: 
        tgs.append(tg)

top_genes = np.array(tgs)
    

cols = 3  
rows = int(np.ceil(num_celltypes / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size as needed

axes = axes.flatten()
# plt.figure(figsize=(15,10))

for idx, ct in enumerate(sel_clust):
	ct_cells = df_umap[df_umap['cluster'] == ct]['cell'].values
	ct_yindxs = np.where(np.isin(p1_ylabel, ct_cells))[0]
	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
	
	df_attn = df_attn.apply(zscore)
	df_attn.fillna(0.0,inplace=True)
	df_attn[df_attn > 5] = 5
	df_attn[df_attn < -5] = -5
	df_attn = df_attn.loc[:,top_genes]
	df_attn = df_attn.loc[top_genes,:]

	print(ct, df_attn.shape)
 
	sns.heatmap(df_attn, cmap='vlag', ax=axes[idx])
	axes[idx].set_title(f"Clustermap for {ct}")

for j in range(idx + 1, rows * cols):
	fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(wdir +cdir+ 'sc_attention_allct.png')
plt.close()


plt.figure(figsize=(15,20))

sel_clust =[
'c_6'
]
df_umap = df_umap.loc[df_umap['cluster'].isin(sel_clust)]

print(df_umap['cluster'].value_counts())

ct= sel_clust[0]
ct_cells = df_umap[df_umap['cluster'] == ct]['cell'].values
ct_yindxs = np.where(np.isin(p1_ylabel, ct_cells))[0]
df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
					index=adata_p1.var.index.values, columns=adata_p1.var.index.values)

df_attn = df_attn.apply(zscore)
df_attn.fillna(0.0,inplace=True)
df_attn[df_attn > 5] = 5
df_attn[df_attn < -5] = -5
df_attn = df_attn.loc[:,top_genes]
df_attn = df_attn.loc[top_genes,:]

print(ct, df_attn.shape)

sns.heatmap(df_attn, cmap='vlag')
plt.tight_layout()
plt.savefig(wdir +cdir+ 'sc_attention_allct_one.png')
plt.close()
