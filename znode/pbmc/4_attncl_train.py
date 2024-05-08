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

sample = 'pbmc'
wdir = 'znode/pbmc/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')

picasa_object = picasa.pic.create_picasa_object({'sc':rna,'sp':spatial},wdir)

params = {'device' : 'cuda',
		'batch_size' : 128,
		'input_dim' : rna.X.shape[1],
		'embedding_dim' : 10000,
		'attention_dim' : 10,
		'latent_dim' : 10,
		'encoder_layers' : [100,10],
		'projection_layers' : [25,25],
		'learning_rate' : 0.001,
		'lambda_attention_sc_entropy_loss' : 1.0,
		'lambda_attention_sp_entropy_loss' : 1.0,
		'lambda_cl_sc_entropy_loss' : 0.5,
		'lambda_cl_sp_entropy_loss' : 0.5,
		'temperature_cl' : 1.0,
		'neighbour_method' : 'exact',
     	'corruption_rate' : 1.0,
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

	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=30,metric='cosine').fit(df_sc)
	df_umap= pd.DataFrame()
	df_umap['cell'] = df_sc.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
	
	dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	dfl.columns = ['cell','celltype','batch']
	df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_sc_latent',pt_size=1.0,ftype='png')

def plot_spatial(label,position,sample_size=2500):
	import random
	import matplotlib.pylab as plt		
	from plotnine import ggplot, geom_point
 

	dfn = pd.DataFrame()
	dfn['x'] = [ float(x.split('x')[0]) for x in position]
	dfn['y'] = [ float(x.split('x')[1]) for x in position]
	dfn['celltype'] = label
	
	sel_indexes = random.sample(range(0,dfn.shape[0]-1), sample_size)
	dfn = dfn.iloc[sel_indexes,:]
	dfn = pd.melt(dfn,id_vars=['x','y'])


	ggplot(dfn, aes(x='x', y='y', color='value')) +geom_point(size=2) 
	plt.savefig('test.png');plt.close()

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

def plot_scsp_overlay():
	import umap
	import h5py as hf
	import random
	from picasa.util.plots import plot_umap_df
	
	picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
	df_sc = pd.DataFrame(picasa_h5['sc_latent'][:],index=rna.obs.index.values)
	df_sp = pd.DataFrame(picasa_h5['sp_latent'][:],index=spatial.obs.index.values)
	picasa_h5.close()

	sc_sel_indxs = np.unique(np.array([ x[1] for x in np.array(picasa_h5['spsc_map']) ]))
	sp_sel_indxs = np.unique(np.array([ x[1] for x in np.array(picasa_h5['scsp_map']) ]))
 
	df_sc = df_sc.iloc[sc_sel_indxs]
	df_sc.index = ['sc_'+x for x in df_sc.index.values]

	df_sp = df_sp.iloc[sp_sel_indxs]
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
	
	from asappy.util.analysis import quantile_normalization
	sc_norm,sp_norm = quantile_normalization(df_sc.to_numpy(),df_sp.to_numpy())
	dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
	dfh.index = dfmain.index.values
    ###################
    ####################
 
	dfh = dfmain

    ###################
    ####################
 
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.8,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	df_umap['batch'] = [x.split('_')[0] for x in df_umap['cell']]
	df_umap['celltype'] = [x.split('_')[1] for x in df_umap['cell']]

 
	dfspl = pd.read_csv(wdir+'data/CytoSPACE_example_colon_cancer_merscope/HumanColonCancerPatient2_ST_celltypes_cytospace.tsv',sep='\t')
	dfspl.columns = ['cell','celltype']
	dfm = pd.DataFrame([x.replace('sp_','') for x in df_umap['cell']],columns=['cell'])
	dfspmerge = pd.merge(dfm,dfspl,on='cell',how='right')
	spmap = {x:y for x,y in zip(dfspmerge['cell'],dfspmerge['celltype'])}
 
	df_umap['celltype'] = [ 'sp_'+spmap[x.replace('sp_','')] if 'sp_' in x else 'sc_'+x.split('_')[1] for x in df_umap['cell'] ]
 
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_scsp_',pt_size=1.0,ftype='png')


# train()
# eval()
# plot_attention()
plot_latent()
# plot_scsp_overlay()

	

# sout = picasa.nn_attn.predict_scsp(picasa_model,x_sc)
# dfsc = pd.DataFrame(sout.h_sc.cpu().detach().numpy(),index=y)
# dfsp = pd.DataFrame(sout.h_spp.cpu().detach().numpy(),index=y)


# umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,metric='cosine').fit(dfsc)
# df_umap= pd.DataFrame()
# df_umap['cell'] = dfsc.index.values
# df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
# df_umap['celltype'] = [x.split('_')[0] for x in df_umap['cell']]
# plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_testsc_',pt_size=1.0,ftype='png')

# umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,metric='cosine').fit(dfsp)
# df_umap= pd.DataFrame()
# df_umap['cell'] = dfsp.index.values
# df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
# df_umap['celltype'] = [x.split('_')[0] for x in df_umap['cell']]
# plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_testsp_',pt_size=1.0,ftype='png')


# dfsc.index = ['sc_'+x for x in dfsc.index.values]
# dfsp.index = ['sp_'+x for x in dfsp.index.values]
# df = pd.concat([dfsc,dfsp])

# umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,metric='cosine').fit(df)
# df_umap= pd.DataFrame()
# df_umap['cell'] = df.index.values
# df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
# df_umap['celltype'] = [x.split('_')[1] for x in df_umap['cell']]
# df_umap['batch'] = [x.split('_')[0] for x in df_umap['cell']]
# plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_testsp_merge_',pt_size=1.0,ftype='png')
# plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_testsp_merge_',pt_size=1.0,ftype='png')
