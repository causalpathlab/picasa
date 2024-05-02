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
	 
		logging.info('Assign neighbour - '+ method)

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
			logging.info('Assign neighbour - manual.')
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

		model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['temperature_cont'],self.wdir+'results/4_attncl_train_loss.txt.gz')

		torch.save(picasa_model.state_dict(),self.wdir+'results/nn_attncl.model')

		logging.info('Completed training...model saved in results/nn_attncl.model')
 
	def eval_model_sc(self,eval_batch_size,device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model'))

		data_pred = dutil.nn_load_data_pairs(self.data.adata_list['sc'],self.data.adata_list['sp'],self.scsp_map,device,eval_batch_size)
  
		df_h_sc = pd.DataFrame()
		df_attn = pd.DataFrame()
		c = 0
		for x_sc,y,x_spp in data_pred:
			m_el,ylabel = model.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
			m = m_el[0]
			df_h_sc = pd.concat([df_h_sc,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
			if c == 0:
				df_attn = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				c += 1
			else:
				df_attn_c = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				df_attn = (df_attn +df_attn_c)/2
	
		df_attn.columns = self.data.adata_list['sc'].var.index.values
		df_attn.index = self.data.adata_list['sc'].var.index.values
		
		self.sc_mean_attention = df_attn
		self.sc_latent = df_h_sc.loc[self.data.adata_list['sc'].obs.index.values,:]
  
	def eval_model_sp(self,eval_batch_size,device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model'))
  
		data_pred = dutil.nn_load_data_pairs(self.data.adata_list['sp'],self.data.adata_list['sc'],self.spsc_map,device,eval_batch_size)
  
		df_h_sp = pd.DataFrame()
		df_attn_sp = pd.DataFrame()
		c = 0
		for x_sc,y,x_spp in data_pred:
			m_el,ylabel = model.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
			m = m_el[0]
			df_h_sp = pd.concat([df_h_sp,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
			if c == 0:
				df_attn_sp = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				c += 1
			else:
				df_attn_c = pd.DataFrame(m.attn_sc.cpu().detach().numpy().mean(0))
				df_attn_sp = (df_attn_sp +df_attn_c)/2
	
		df_attn_sp.columns = self.data.adata_list['sp'].var.index.values
		df_attn_sp.index = self.data.adata_list['sp'].var.index.values
		
		self.sp_mean_attention = df_attn_sp
		self.sp_latent = df_h_sp.loc[self.data.adata_list['sp'].obs.index.values,:]
  
	def save(self):
		import h5py
		with h5py.File(self.wdir+'results/picasa_out.h5', 'w') as f:
			for attr_name, attr_value in vars(self).items():
				if isinstance(attr_value, pd.DataFrame):
					f.create_dataset(attr_name, data=attr_value)
				elif isinstance(attr_value, dict) and '_map' in attr_name:
					df = pd.DataFrame(attr_value.items())
					f.create_dataset(attr_name, data=df)

	def plot_loss(self):
		from picasa.util.plots import plot_loss
		plot_loss(self.wdir+'results/4_attncl_train_loss.txt.gz',self.wdir+'results/4_attncl_train_loss.png')

def create_picasa_object(adata_list: Adata,wdir: str):
	return picasa(dutil.data.Dataset(adata_list),wdir)
