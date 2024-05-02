from .util.typehint import Adata
from . import dutil 
from . import model 
from . import neighbour
import torch
import logging
import pandas as pd
import numpy as np


class picasa(object):
	def __init__(self, data: dutil.data.Dataset, wdir: str):
		self.data = data
		self.wdir = wdir
		logging.basicConfig(filename=self.wdir+'results/4_attncl_train.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
	
	def estimate_neighbour(self,method='approx_50'):
		if 'approx' in method :
			number_of_trees = int(method.split('_')[1])
			self.scsp_map = neighbour.generate_neighbours(self.data.adata_list['sp'],self.data.adata_list['sc'],'scsp',number_of_trees)
			self.spsc_map = neighbour.generate_neighbours(self.data.adata_list['sc'],self.data.adata_list['sp'],'spsc',number_of_trees)
		elif method == 'exact':
			from scipy.spatial.distance import cdist
			logging.info('Generating neighbour list using exact method - cdist...')
			distmat =  cdist(self.data.adata_list['sp'].X.todense(), self.data.adata_list['sc'].X.todense())
			sorted_indices_sp = np.argsort(distmat, axis=1)
			sorted_indices_sc = np.argsort(distmat.T, axis=1)
			self.scsp_map = {x:y[0] for x,y in enumerate(sorted_indices_sc)}
			self.spsc_map = {x:y[0] for x,y in enumerate(sorted_indices_sp)}
			logging.info('Exact neighbour estimate complete.')

			
	def assign_neighbour(self, scsp_map: dict, spsc_map: dict):
			logging.info('Assigned neighbour - manual.')
			self.scsp_map = scsp_map
			self.spsc_map = spsc_map
	
	def set_nn_params(self,params: dict):
		self.nn_params = params
	
	def train(self):

		logging.info('Starting training...')

		logging.info(self.nn_params)
  
		data = dutil.nn_load_data_pairs(self.data.adata_list['sc'],self.data.adata_list['sp'],self.scsp_map,self.nn_params['device'],self.nn_params['batch_size'])

		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss']).to(self.nn_params['device'])
  
		logging.info(picasa_model)

		model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['temperature_cont'],self.wdir+'results/4_attncl_train_loss.txt')

		torch.save(picasa_model.state_dict(),self.wdir+'results/nn_attncl.model')

		logging.info('Completed training...model saved in results/nn_attncl.model')
 
	def load_model(self,eval_batch_size,device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model'))

		data_pred = dutil.nn_load_data_pairs(self.data.adata_list['sc'],self.data.adata_list['sp'],self.scsp_map,device,eval_batch_size)
  

		df_h_sc = pd.DataFrame()
		df_attn = pd.DataFrame()
		c = 0
		for x_sc,y,x_spp in data_pred:
			m,ylabel = model.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
			df_h_sc = pd.concat([df_h_sc,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
			if c == 0:
				df_attn = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				c += 1
			else:
				df_attn_c = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				df_attn = (df_attn +df_attn_c)/2
	
		df_attn.columns = self.data.adata_list['sc'].var.index.values
		df_attn.index = self.data.adata_list['sc'].var.index.values
		
		self.data.adata_list['sc'].uns = {}
		self.data.adata_list['sc'].uns['mean_attention'] = df_attn
		self.data.adata_list['sc'].uns['h_sc'] = df_h_sc.loc[self.data.adata_list['sc'].obs.index.values,:]
		
def create_picasa_object(adata_list: Adata,wdir: str):
	return picasa(dutil.data.Dataset(adata_list),wdir)
