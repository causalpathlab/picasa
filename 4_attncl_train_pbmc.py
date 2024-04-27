

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch
import sys
import logging

sample = 'colon'
wdir = 'znode/colon/'
# sample = 'pbmc'
# wdir = 'znode/pbmc/'

rna = an.read_h5ad(wdir+'data/'+sample+'_sc.h5ad')
spatial = an.read_h5ad(wdir+'data/'+sample+'_sp.h5ad')
distdf = pd.read_csv(wdir+'data/sc_sp_dist.csv.gz')


device = 'cuda'
batch_size = 256
eval_batch_size= int(rna.X.shape[0]/5)
input_dims = rna.X.shape[1]
emb_dim = 3000
att_dim = 10
latent_dims = 10
encoder_layers = [200,100,10]
projection_layers = [10,25,25]
l_rate = 0.001
epochs= 1000

temperature = 1.0 # higher scores smooths out the differences between the similarity scores.



logging.basicConfig(filename=wdir+'results/4_attncl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

logging.info( f"Device: {device}, \
	Batch Size: {batch_size}, Evaluation Batch Size: {eval_batch_size}, \
	Input Dimensions: {input_dims}, Latent Dimensions: {latent_dims}, \
	Encoder Layers: {encoder_layers}, Projection Layers: {projection_layers}, \
	Learning Rate: {l_rate}, \
	Epochs: {epochs}, temperature: {temperature}")

def train():
	logging.info('train...')
	data = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,batch_size)

	sailr_model = sailr.nn_attn.SAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	logging.info(sailr_model)

	sailr.nn_attn.train(sailr_model,data,epochs,l_rate,temperature)

	torch.save(sailr_model.state_dict(),wdir+'results/nn_attncl.model')
 
def train_mgpu():
	logging.info('train...')
	data = sailr.du.dataloader_pair.nn_load_data_mgpu(rna,spatial,distdf,device,batch_size)
 
	train_dataloader = data.train_dataloader()

	lossf = wdir+'results/nn_attncl_model_loss.txt'
	sailr_model = sailr.nn_attn.LitSAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers,temperature,lossf)
	sailr_model.to(device)
 
	logging.info(sailr_model)
 
	trainer = sailr.nn_attn.pl.Trainer(
	max_epochs=epochs,
	accelerator='gpu',
	plugins= sailr.nn_attn.DDPPlugin(find_unused_parameters=False),
	gradient_clip_val=0.5,
	progress_bar_refresh_rate=50,
	enable_checkpointing=False)


	trainer.fit(sailr_model,train_dataloader)
 
	torch.save(sailr_model.state_dict(),wdir+'results/nn_attncl.model')

def eval():
	
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	
	device = 'cpu'
	data_pred = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,eval_batch_size)

	sailr_model = sailr.nn_attn.SAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	m,ylabel = sailr.nn_attn.predict(sailr_model,data_pred)


	# dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
	# dfl.columns = ['cell','celltype','batch']

	# dflv2 = dfl.loc[dfl['cell'].str.contains('V2'),:]
	# selected_index = dflv2.loc[dflv2['celltype'] == "CD14+ monocyte"].index.values
	# selected_index = dflv2.loc[dflv2['celltype'] == "CD4+ T cell"].index.values

	attn_mtx = m.attn_sc.cpu().detach().numpy()
	selected_index = [x for x,y in enumerate(ylabel) if 'T_' in y]
	df_attn = pd.DataFrame(attn_mtx.mean(0))
	# df_attn = pd.DataFrame(attn_mtx[selected_index].mean(0))
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
	plt.savefig(wdir+'results/attn_markert_mono.png')
	plt.close()
	sns.clustermap(df_attn,cmap='viridis')
	plt.savefig(wdir+'results/attn_markert_monoc.png')
	plt.close()


	def plot_latent(mtx,label):
		# mtx = m.h_sc.cpu().detach().numpy();label='h_sc'
		dfh = pd.DataFrame(mtx)
		umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.6,metric='cosine').fit(dfh)

		df_umap= pd.DataFrame()
		df_umap['cell'] = ylabel
		df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


		# df_umap['celltype'] = [x.split('_')[2] for x in df_umap['cell']]
		# df_umap['celltype'] = rna.obs.celltype.values
		# df_umap['celltype'] = pd.merge(df_umap,rna.obs,left_on='cell',right_index=True,how='left')['celltype'].values


	
		# df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values
		df_umap['celltype'] = [x.split('_')[0] for x in df_umap['cell']]


		plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_'+label,pt_size=1.0,ftype='png')


	plot_latent(m.h_sc.cpu().detach().numpy(),'h_sc')
	plot_latent(m.h_spp.cpu().detach().numpy(),'h_spp')

	# plot_latent(m.z_sc.cpu().detach().numpy(),'z_sc')
	# plot_latent(m.z_spp.cpu().detach().numpy(),'z_spp')

def eval2(rna,spatial,distdf):
    
	import umap
	from sailr.util.plots import plot_umap_df
	
	logging.info('eval...')
	device = 'cpu'
	data_pred = sailr.du.nn_load_data_pairs(rna,spatial,distdf,device,int(rna.shape[0]/10))
	sailr_model = sailr.nn_attn.SAILRNET(input_dims, emb_dim,att_dim,latent_dims, encoder_layers, projection_layers).to(device)
	sailr_model.load_state_dict(torch.load(wdir+'results/nn_attncl.model'))
	m,ylabel = sailr.nn_attn.predict(sailr_model,data_pred)
 
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
 
	data_pred = sailr.du.nn_load_data_pairs(spatial_eval,rna,distdf,device,int(spatial.shape[0]/1))
	m,ylabel = sailr.nn_pcl.predict(sailr_model,data_pred)
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
# train()
# train_mgpu()
eval() 