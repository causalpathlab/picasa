
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
distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')

device = 'cuda'
batch_size = 128
input_dims = rna.X.shape[1]
emb_dim = 3000
att_dim = 10
latent_dims = 10
encoder_layers = [200,100,10]
projection_layers = [10,25,25]
l_rate = 0.001
epochs= 100

temperature = 1.0 # higher scores smooths out the differences between the similarity scores.



logging.basicConfig(filename=wdir+'results/4_attncl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

logging.info( f"Device: {device}, \
	Batch Size: {batch_size}, \
	Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
	Encoder Layers: {encoder_layers}, Projection Layers: {projection_layers}, \
	Learning Rate: {l_rate}, \
	Epochs: {epochs}, temperature: {temperature}")

def train():
	logging.info('train...')
	data = picasa.du.nn_load_data_pairs(rna,spatial,distdf,device,batch_size)

	picasa_model = picasa.nn_attn.PICASANET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	logging.info(picasa_model)

	picasa.nn_attn.train(picasa_model,data,epochs,l_rate,temperature)

	torch.save(picasa_model.state_dict(),wdir+'results/nn_attncl.model')
 

def eval():
	
	import umap
	from picasa.util.plots import plot_umap_df
	
	logging.info('eval...')
	
	device = 'cpu'
	batch= 5
	eval_batch_size = rna.shape[0]
	data_pred = picasa.du.nn_load_data_pairs(rna,spatial,distdf,device,eval_batch_size)

	picasa_model = picasa.nn_attn.PICASANET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	picasa_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	m,ylabel = picasa.nn_attn.predict(picasa_model,data_pred)



	dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	dfl.columns = ['cell','celltype','batch']


	attn_mtx = m.attn_sc.cpu().detach().numpy()
	df_attn = pd.DataFrame(attn_mtx.mean(0))
	df_attn.columns = rna.var.index.values
	df_attn.index = rna.var.index.values
	
	# sg = ["IL7R", "CCR7","CD14", "LYZ", "S100A4","MS4A1","CD8A","FCGR3A", "MS4A7","GNLY", "NKG7","FCER1A", "CST3"]
	sg = [ "CCR7","CD14", "LYZ","MS4A1","CD8A","FCGR3A","GNLY", "NKG7","FCER1A"]

	# sg = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14', 'LGALS3',
    #    'S100A8', 'GNLY', 'NKG7', 'KLRB1', 'FCGR3A', 'MS4A7', 'FCER1A',
    #    'CST3', 'CD3D', 'CD27', 'SELL', 'CCR7', 'IL32', 'GZMA', 'GZMK',
    #    'DUSP2', 'GZMH', 'GZMB', 'CD79B', 'CD86']
 
 
	df_attn = df_attn.loc[sg,sg] 
	from scipy.stats import zscore
	zscore_df = df_attn.apply(zscore)
	
	sns.heatmap(zscore_df,cmap='viridis')
	plt.savefig(wdir+'results/attn_markert_test.png')
	plt.close()
	sns.heatmap(df_attn,cmap='viridis')
	plt.savefig(wdir+'results/attn_markert_test_df.png')
	plt.close()
	
	zscore_df[zscore_df>5]=5
	zscore_df[zscore_df<-5]=-5
	sns.clustermap(zscore_df,cmap='viridis')
	plt.savefig(wdir+'results/attn_markert_test_df_clust.png')
	plt.close()


	def plot_latent(mtx,label):
		dfh = pd.DataFrame(mtx)
		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.6,metric='cosine').fit(dfh)

		df_umap= pd.DataFrame()
		df_umap['cell'] = ylabel
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values


		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_'+label,pt_size=1.0,ftype='png')


	plot_latent(m.h_sc.cpu().detach().numpy(),'h_sc')
	plot_latent(m.h_spp.cpu().detach().numpy(),'h_spp')

	# plot_latent(m.z_sc.cpu().detach().numpy(),'z_sc')
	# plot_latent(m.z_spp.cpu().detach().numpy(),'z_spp')

def eval2(rna,spatial,distdf):
    
	import umap
	from picasa.util.plots import plot_umap_df
	
	logging.info('eval...')
	device = 'cpu'
	data_pred = picasa.du.nn_load_data_pairs(rna,spatial,distdf,device,int(rna.shape[0]/10))
	picasa_model = picasa.nn_attn.PICASANET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	picasa_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	m,ylabel = picasa.nn_attn.predict(picasa_model,data_pred)
 
	dfsc = pd.DataFrame(m.h_sc.cpu().detach().numpy())
	dfsc.index = ['sc_'+x for x in ylabel]

	# dfsp = pd.DataFrame(m.h_spp.cpu().detach().numpy())
	# sc_index = pd.merge(dfsc,rna.obs.reset_index().reset_index(),left_index=True,right_on='index')['level_0'].values
	# sp_index = distdf.iloc[sc_index]['0'].values
	# dfsp.index = [x.replace('sp_','') for x in spatial.obs.index.values[sp_index]]

	from scipy.spatial.distance import cdist

	sp_index = distdf['0'].unique()
	spatial_eval = spatial[sp_index,:]
	distmat =  cdist(rna.X.todense(), spatial_eval.X.todense())
	distmat = distmat.T
	sorted_indices = np.argsort(distmat, axis=1)
	distdf = pd.DataFrame(sorted_indices)

	distdf = distdf[[0,distdf.shape[1]-1]]
 
	data_pred = picasa.du.nn_load_data_pairs(spatial_eval,rna,distdf,device,int(spatial.shape[0]/1))
	m,ylabel = picasa.nn_pcl.predict(picasa_model,data_pred)
	dfsp = pd.DataFrame(m.h_sc.cpu().detach().numpy())
	dfsp.index = ['sp_ct_'+x for x in ylabel]
 
	

	dfmain = pd.concat([dfsc,dfsp])

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
    
    ######
    
	from asappy.util.analysis import quantile_normalization
	sc_norm,sp_norm = quantile_normalization(dfsc.to_numpy(),dfsp.to_numpy())
	dfh = pd.DataFrame(np.concatenate([sc_norm, sp_norm], axis=0))
 
    ###################
    ####################
 
	dfh.index = dfmain.index.values
	dfh = dfmain
	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.8,metric='cosine').fit(dfh)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfh.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	df_umap['batch'] = [x.split('_')[0] for x in df_umap['cell']]
	df_umap['celltype'] = [x.split('_')[1] for x in df_umap['cell']]

	# dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	# dfl.columns = ['cell','celltype','batch']
	# df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values
 
	plot_umap_df(df_umap,'batch',wdir+'results/nn_pcl_3',pt_size=1.0,ftype='png')
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_pcl_3',pt_size=1.0,ftype='png')
 
	dfspl = pd.read_csv(wdir+'data/CytoSPACE_example_colon_cancer_merscope/HumanColonCancerPatient2_ST_celltypes_cytospace.tsv',sep='\t')
	dfspl.columns = ['cell','celltype']
	dfm = pd.DataFrame(ylabel,columns=['cell'])
	dfspmerge = pd.merge(dfm,dfspl,on='cell',how='left')
	spmap = {x:y for x,y in zip(dfspmerge['cell'],dfspmerge['celltype'])}
 
	df_umap['celltype'] = [ spmap[x.replace('sp_ct_','')] if 'sp_' in x else x.split('_')[1] for x in df_umap['cell'] ]
 
	
	plot_umap_df(df_umap.loc[df_umap['batch']=='sc',:],'celltype',wdir+'results/n n_pcl_sc_',pt_size=1.0,ftype='png')
	plot_umap_df(df_umap.loc[df_umap['batch']=='sp',:],'celltype',wdir+'results/n n_pcl_sp_',pt_size=1.0,ftype='png')
 
def eval3():
	
	import umap
	from picasa.util.plots import plot_umap_df
	
	logging.info('eval...')
	
	device = 'cpu'
	batch= 5
	eval_batch_size = int(rna.shape[0]/batch)
 
	picasa_model = picasa.nn_attn.PICASANET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	picasa_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	
	data_pred = picasa.du.nn_load_data_pairs(rna,spatial,distdf,device,eval_batch_size)

	dfsc = pd.DataFrame()
	for x_sc,y,x_spp in data_pred:
		m,ylabel = picasa.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
		dfsc = pd.concat([dfsc,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
		print(dfsc.shape)

	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,metric='cosine').fit(dfsc)

	df_umap= pd.DataFrame()
	df_umap['cell'] = dfsc.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

	df_umap['celltype'] = [x.split('_')[0] for x in df_umap['cell']]

	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_allsc_',pt_size=1.0,ftype='png')

train()
eval() 

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
