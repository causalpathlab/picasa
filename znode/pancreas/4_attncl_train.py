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

sample = 'pancreas'
wdir = 'znode/pancreas/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')

picasa_object = picasa.pic.create_picasa_object({'sc':rna,'sp':spatial},wdir)

params = {
        'device' : 'cuda',
		'batch_size' : 128,
		'input_dim' : rna.X.shape[1],
		'embedding_dim' : 2000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [25,25],
		'learning_rate' : 0.01,
		'lambda_attention_sc_entropy_loss' : 1.0,
		'lambda_attention_sp_entropy_loss' : 1.0,
		'lambda_cl_sc_entropy_loss' : 0.5,
		'lambda_cl_sp_entropy_loss' : 0.5,
		'temperature_cl' : 1.0,
		'neighbour_method' : 'exact',
     	'corruption_rate' : 0.0,
		'epochs': 100
		}  


def train():
    
	# distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')
	# scsp_map = {x:y[0] for x,y in enumerate(distdf.values)}
	# picasa_object.assign_neighbour(scsp_map,None)
	
	picasa_object.estimate_neighbour(params['neighbour_method'])
	
	picasa_object.set_nn_params(params)
	picasa_object.train()
	picasa_object.plot_loss()

def eval():
	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = int(rna.shape[0]/5)
	picasa_object.eval_model_sc(eval_batch_size,device)
	picasa_object.eval_model_sp(eval_batch_size,device)
	picasa_object.save()

def plot_latent():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	df_sc = pd.DataFrame(picasa_h5['sc_latent'][:],index=rna.obs.index.values)
	picasa_h5.close()

	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=1.0, n_neighbors=30,metric='cosine').fit(df_sc)
	df_umap= pd.DataFrame()
	df_umap['cell'] = df_sc.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	# from sklearn.manifold import TSNE
	# tsne_2d = TSNE(n_components=2, init='random', random_state=0, perplexity=15, metric='cosine')
	# tsne_embedding = tsne_2d.fit_transform(df_sc)
	# df_tsne = pd.DataFrame()
	# df_tsne['cell'] = df_sc.index.values
	# df_tsne[['tsne1', 'tsne2']] = tsne_embedding[:, [0, 1]]
	# df_tsne.rename(columns={'tsne1':'umap1','tsne2':'umap2'},inplace=True)
	# df_umap = df_tsne
 
 
	df_umap['celltype'] = rna.obs.loc[df_sc.index.values,:]['celltype'].values
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_sc_latent',pt_size=1.0,ftype='png')

def plot_attention():

	import h5py as hf
	from scipy.stats import zscore
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	df_attn = pd.DataFrame(picasa_h5['sc_mean_attention'][:],index=rna.var.index.values,columns=rna.var.index.values)
	picasa_h5.close()

	zscore_df = df_attn.apply(zscore)
		
	zscore_df[zscore_df>5]=5
	zscore_df[zscore_df<-5]=-5
	sns.clustermap(zscore_df,cmap='viridis')
	plt.savefig(wdir+'results/sc_attention.png')
	plt.close()
	sns.heatmap(zscore_df,cmap='viridis')
	plt.savefig(wdir+'results/sc_attention_hmap.png')
	plt.close()
	

def plot_scsp_overlay():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	df_sc = pd.DataFrame(picasa_h5['sc_latent'][:],index=rna.obs.index.values)
	df_sp = pd.DataFrame(picasa_h5['sp_latent'][:],index=spatial.obs.index.values)

	# sc_sel_indxs = np.unique(np.array([ x[1] for x in np.array(picasa_h5['spsc_map']) ]))
	# sp_sel_indxs = np.unique(np.array([ x[1] for x in np.array(picasa_h5['scsp_map']) ]))
 
	picasa_h5.close()

	# df_sc = df_sc.iloc[sc_sel_indxs]
	df_sc.index = ['sc_'+x for x in df_sc.index.values]

	# df_sp = df_sp.iloc[sp_sel_indxs]
	df_sp.index = ['sp_'+x for x in df_sp.index.values]

	dfmain = pd.concat([df_sc,df_sp])

    ###################
    ####################

	## use std norm or quant norm 
	# from sklearn.preprocessing import StandardScaler
	# def standardize_row(row):
	# 	scaler = StandardScaler()
	# 	row_reshaped = row.values.reshape(-1, 1)  
	# 	row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
	# 	return pd.Series(row_standardized, index=row.index)
	# dfh = dfmain.apply(standardize_row, axis=1)
	# dfh.index = dfmain.index.values
    ######
	
	# from asappy.util.analysis import quantile_normalization
	# sc_norm,sp_norm = quantile_normalization(df_sc.to_numpy(),df_sp.to_numpy())
	# dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
	# dfh.index = dfmain.index.values
    ###################
    ####################
 
	dfh = dfmain

    ###################
    ####################
 
	# umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=1.0, n_neighbors=30,metric='cosine').fit(dfh)

	# df_umap= pd.DataFrame()
	# df_umap['cell'] = dfh.index.values
	# df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	from sklearn.manifold import TSNE

	tsne_2d = TSNE(n_components=2, init='random', random_state=0, perplexity=10, metric='euclidean')
	tsne_embedding = tsne_2d.fit_transform(dfh)
	df_tsne = pd.DataFrame()
	df_tsne['cell'] = dfh.index.values
	df_tsne[['tsne1', 'tsne2']] = tsne_embedding[:, [0, 1]]
	df_tsne.rename(columns={'tsne1':'umap1','tsne2':'umap2'},inplace=True)
	df_umap = df_tsne
 
	dfl = pd.read_csv(wdir+'data/pancreas_label.csv.gz')
	dfl.columns = ['cell','batch','celltype']
 
	dfm = pd.DataFrame([x.replace('sp_','').replace('sc_','') for x in df_umap['cell']],columns=['cell'])
	dfspmerge = pd.merge(dfm,dfl,on='cell',how='right')
	ct_map = {x:y for x,y in zip(dfspmerge['cell'],dfspmerge['celltype'])} 
	b_map = {x:y for x,y in zip(dfspmerge['cell'],dfspmerge['batch'])}
 
	df_umap['celltype'] = [ 'sp_'+ct_map[x.replace('sp_','')] if 'sp_' in x else 'sc_'+ct_map[x.replace('sc_','')] for x in df_umap['cell'] ]
	df_umap['batch'] = [ 'sp_'+b_map[x.replace('sp_','')] if 'sp_' in x else 'sc_'+b_map[x.replace('sc_','')] for x in df_umap['cell'] ]
 
 
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')
 
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')

	df_umap['celltype2'] = [ x.replace('sp_','').replace('sc_','') for x in df_umap['celltype'] ]
 
	plot_umap_df(df_umap,'celltype2',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')


train()
eval()
plot_attention()
plot_latent()
plot_scsp_overlay()

	