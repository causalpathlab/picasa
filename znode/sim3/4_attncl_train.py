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

sample = 'sim3'
wdir = 'znode/sim3/'

batch1 = an.read_h5ad(wdir+'data/'+sample+'_batch1.h5ad')
batch2 = an.read_h5ad(wdir+'data/'+sample+'_batch2.h5ad')
batch3 = an.read_h5ad(wdir+'data/'+sample+'_batch3.h5ad')

picasa_object = picasa.pic.create_picasa_object(
    {'batch1':batch1,
     'batch2':batch2,
     'batch3':batch3
     },
    wdir)

params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch1.X.shape[1],
		'embedding_dim' : 10000,
		'attention_dim' : 10,
		'latent_dim' : 10,
		'encoder_layers' : [100,10],
		'projection_layers' : [10,10],
		'learning_rate' : 0.001,
		'lambda_attention_sc_entropy_loss' : 1.0,
		'lambda_attention_sp_entropy_loss' : 1.0,
		'lambda_cl_sc_entropy_loss' : 0.5,
		'lambda_cl_sp_entropy_loss' : 0.5,
		'temperature_cl' : 1.0,
		'neighbour_method' : 'approx_50',
     	'corruption_rate' : 0.0,
		'epochs': 1,
		'titration': 10
		}  


def train():
    
	# distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')
	# scsp_map = {x:y[0] for x,y in enumerate(distdf.values)}
	# picasa_object.assign_neighbour(scsp_map,None)
	
	picasa_object.estimate_neighbour(params['neighbour_method'])
	
	picasa_object.set_nn_params(params)
	picasa_object.train(params['titration'])
	# picasa_object.plot_loss()

def eval():
	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = int(batch1.shape[0]/5)
	picasa_object.eval_model(eval_batch_size,device)
	picasa_object.save()

def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	dfl = pd.read_csv(wdir+'data/sim3_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	for batch in batch_keys:
		df = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])

		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.8,n_neighbors=30,metric='cosine').fit(df)
		df_umap= pd.DataFrame()
		df_umap['cell'] = df.index.values
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		# from sklearn.manifold import TSNE
		# tsne_2d = TSNE(n_components=2, init='random', random_state=0, perplexity=15, metric='cosine')
		# tsne_embedding = tsne_2d.fit_transform(df_sc)
		# df_tsne = pd.DataFrame()
		# df_tsne['cell'] = df_sc.index.values
		# df_tsne[['tsne1', 'tsne2']] = tsne_embedding[:, [0, 1]]
		# df_tsne.rename(columns={'tsne1':'umap1','tsne2':'umap2'},inplace=True)
		# df_umap = df_tsne
	
	
		df_umap['celltype'] = pd.merge(df_umap,dfl.loc[dfl['batch']==batch],on='cell',how='left')['celltype'].values
		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')

		df_umap['celltype2'] = [ x.split('-')[0] for x in df_umap['celltype']]
		plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_lat_'+batch,pt_size=1.0,ftype='png')
	picasa_h5.close()

def plot_attention():

	import h5py as hf
	from scipy.stats import zscore
	
	### mean 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	df_attn = pd.DataFrame(np.mean(picasa_h5['sc_attention'],axis=0),index=batch1.var.index.values,columns=batch1.var.index.values)
	picasa_h5.close()
	zscore_df = df_attn.apply(zscore)
	zscore_df[zscore_df>5]=5
	zscore_df[zscore_df<-5]=-5
	sns.clustermap(zscore_df,cmap='viridis')
	plt.savefig(wdir+'results/sc_attention_mean.png')
	plt.close()
	
	### celltype 
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	ylabel = np.array([x.decode('utf-8') for x in picasa_h5['sc_ylabel']])

	unique_celltypes = batch1.obs['celltype'].unique()
	num_celltypes = len(unique_celltypes)
	top_genes = []
	top_n = 5
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5['sc_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5

		df_attn = df_attn.unstack().reset_index()
		df_attn = df_attn[df_attn['level_0'] != df_attn['level_1']]
		df_attn = df_attn.sort_values(0,ascending=False)
		top_genes.append(df_attn['level_0'].unique()[:top_n])
	top_genes = np.unique(np.array(top_genes).flatten())
 
 
	cols = 3  
	rows = int(np.ceil(num_celltypes / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size as needed

	axes = axes.flatten()
  
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5['sc_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5
		df_attn = df_attn.loc[:,top_genes]
		df_attn = df_attn.loc[top_genes,:]

		sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
		axes[idx].set_title(f"Clustermap for {ct}")

	for j in range(idx + 1, rows * cols):
		fig.delaxes(axes[j])

	plt.tight_layout()
	plt.savefig(wdir + 'results/sc_attention_allct.png')
	plt.close()
 
	for idx, ct in enumerate(unique_celltypes):
		ct_ylabel = batch1.obs[batch1.obs['celltype'] == ct].index.values
		ct_yindxs = np.where(np.isin(ylabel, ct_ylabel))[0]
		df_attn = pd.DataFrame(np.mean(picasa_h5['sc_attention'][ct_yindxs], axis=0),
							index=batch1.var.index.values, columns=batch1.var.index.values)
		
		df_attn = df_attn.apply(zscore)
		df_attn[df_attn > 5] = 5
		df_attn[df_attn < -5] = -5

		# df = df_attn.unstack().reset_index()
		# df = df[df['level_0'] != df['level_1']]
		# df = df.sort_values(0,ascending=False)
		# tg = df['level_0'].unique()[:top_n]
		# df_attn = df_attn.loc[:,tg]
		# df_attn = df_attn.loc[tg,:]

		sns.clustermap(df_attn, cmap='viridis')
		plt.tight_layout()
		plt.savefig(wdir + 'results/sc_attention_'+ct+'.png')
		plt.close()

def plot_scsp_overlay():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]
	
	dfmain = pd.DataFrame()
	for batch in batch_keys:
		df_c = pd.DataFrame(picasa_h5[batch+'_latent'][:],index=[x.decode('utf-8') for x in picasa_h5[batch+'_ylabel']])
		df_c.index = [batch+'_'+str(x) for x in df_c.index.values]
		dfmain = pd.concat([dfmain,df_c],axis=0)
  
	picasa_h5.close()

    ###################
    ####################

	# use std norm or quant norm 
	# from sklearn.preprocessing import StandardScaler
	# def standardize_row(row):
	# 	scaler = StandardScaler()
	# 	row_reshaped = row.values.reshape(-1, 1)  
	# 	row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
	# 	return pd.Series(row_standardized, index=row.index)
	# dfh = dfmain.apply(standardize_row, axis=1)
	# dfh.index = dfmain.index.values
    # ######
	
	# from asappy.util.analysis import quantile_normalization
	# sc_norm,sp_norm = quantile_normalization(df_sc.to_numpy(),df_sp.to_numpy())
	# dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
	# dfh.index = dfmain.index.values
    ###################
    ####################
 
	dfh = dfmain

    ###################
    ####################
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=30,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	# from sklearn.manifold import TSNE
	# tsne_2d = TSNE(n_components=2, init='random', random_state=0, perplexity=10, metric='euclidean')
	# tsne_embedding = tsne_2d.fit_transform(dfh)
	# df_tsne = pd.DataFrame()
	# df_tsne['cell'] = dfh.index.values
	# df_tsne[['tsne1', 'tsne2']] = tsne_embedding[:, [0, 1]]
	# df_tsne.rename(columns={'tsne1':'umap1','tsne2':'umap2'},inplace=True)
	# df_umap = df_tsne
 
	dfl = pd.read_csv(wdir+'data/sim3_label.csv.gz')
	dfl.columns = ['index','cell','batch','celltype']
	dfl['cell'] = [x +'_'+y for x,y in zip(dfl['batch'],dfl['cell'])]
 

	pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
 
	df_umap['celltype'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['celltype'].values
	df_umap['batch'] = pd.merge(df_umap['cell'],dfl,on='cell',how='left')['batch'].values
	df_umap['celltype2'] = [ x.split('-')[0] for x in df_umap['celltype']]
 
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 
	plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png') 



train()
eval()
plot_latent()
# plot_scsp_overlay()
# plot_attention()

	