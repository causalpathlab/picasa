from .util.typehint import Adata
from . import dutil 
from . import model 
from . import neighbour
import torch
import logging
import pandas as pd
import numpy as np
import itertools

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
		self.nbr_map = {}
		self.adata_pairs = list(itertools.combinations(range(len(self.data.adata_list)), 2))	
		self.adata_keys = list(self.data.adata_list.keys())
  
		if 'approx' in method :

			number_of_trees = int(method.split('_')[1])
   
			for ad_pair in self.adata_pairs:
				p1 = self.adata_keys[ad_pair[0]]
				p2 = self.adata_keys[ad_pair[1]]
				logging.info(str(self.data.adata_list[p1].X.shape))
				logging.info(str(self.data.adata_list[p2].X.shape))
				self.nbr_map[p1+'_'+p2] = neighbour.generate_neighbours(self.data.adata_list[p2],self.data.adata_list[p1],p1+p2,number_of_trees)
				self.nbr_map[p2+'_'+p1] = neighbour.generate_neighbours(self.data.adata_list[p1],self.data.adata_list[p2],p2+p1,number_of_trees)
    
		elif method == 'exact':
			from scipy.spatial.distance import cdist
			logging.info('Generating neighbour list using exact method - cdist...')
      
			for ad_pair in self.adata_pairs:
				p1 = self.adata_keys[ad_pair[0]]
				p2 = self.adata_keys[ad_pair[1]]

				logging.info(str(self.data.adata_list[p1].X.shape))
				logging.info(str(self.data.adata_list[p2].X.shape))
				distmat =  cdist(self.data.adata_list[p1].X.todense(), self.data.adata_list[p2].X.todense())
				sorted_indices_p1 = np.argsort(distmat, axis=1)
				sorted_indices_p2 = np.argsort(distmat.T, axis=1)
				self.nbr_map[p2+'_'+p1] = {x:y[0] for x,y in enumerate(sorted_indices_p2)}
				self.nbr_map[p1+'_'+p2] = {x:y[0] for x,y in enumerate(sorted_indices_p1)}
			logging.info('Exact neighbour estimate complete.')

				
	def set_nn_params(self,params: dict):
		self.nn_params = params
	
	def train(self):

		logging.info('Starting training...')

		logging.info(self.nn_params)
  
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss'],self.nn_params['corruption_rate']).to(self.nn_params['device'])
  
		logging.info(picasa_model)
  
		for ad_pair in self.adata_pairs:
			p1 = self.adata_keys[ad_pair[0]]
			p2 = self.adata_keys[ad_pair[1]]
   
			logging.info('Training...model-'+p1+'_'+p2)
	
			data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],self.nn_params['device'],self.nn_params['batch_size'])

			model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['temperature_cl'],self.wdir+'results/4_attncl_train_loss'+p1+'_'+p2+'.txt.gz')
   
			logging.info('Training...model-'+p2+'_'+p1)
   
			data = dutil.nn_load_data_pairs(self.data.adata_list[p2],self.data.adata_list[p1],self.nbr_map[p2+'_'+p1],self.nn_params['device'],self.nn_params['batch_size'])

			model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['temperature_cl'],self.wdir+'results/4_attncl_train_loss'+p2+'_'+p1+'.txt.gz')

		torch.save(picasa_model.state_dict(),self.wdir+'results/nn_attncl.model')

		logging.info('Completed training...model saved in results/nn_attncl.model')
 
	def eval_model(self,eval_batch_size,device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'],self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['lambda_attention_sc_entropy_loss'],self.nn_params['lambda_attention_sp_entropy_loss'],self.nn_params['lambda_cl_sc_entropy_loss'],self.nn_params['lambda_cl_sp_entropy_loss'],self.nn_params['corruption_rate']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model'))
		self.attention = {}
		self.latent = {}
		self.ylabel = {}

		for ad_pair in self.adata_pairs:
      
			p1 = self.adata_keys[ad_pair[0]]
			p2 = self.adata_keys[ad_pair[1]]
   

			data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],device,eval_batch_size)
	
			df_h_sc = pd.DataFrame()
			attn_list = []

			for x_sc,y,x_spp in data_pred:
				m_el,ylabel = model.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
				m = m_el[0]
				df_h_sc = pd.concat([df_h_sc,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
				attn_list.append(m.attn_sc.cpu().detach().numpy())
	
			self.attention[p1] = np.concatenate(attn_list, axis=0)
			self.latent[p1] = df_h_sc
			self.ylabel[p1] = df_h_sc.index.values
   
			data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p2],self.data.adata_list[p1],self.nbr_map[p2+'_'+p1],device,eval_batch_size)
	
			df_h_sc = pd.DataFrame()
			attn_list = []

			for x_sc,y,x_spp in data_pred:
				m_el,ylabel = model.nn_attn.predict_batch(picasa_model,x_sc,y,x_spp)
				m = m_el[0]
				df_h_sc = pd.concat([df_h_sc,pd.DataFrame(m.h_sc.cpu().detach().numpy(),index=ylabel)],axis=0)
				attn_list.append(m.attn_sc.cpu().detach().numpy())
	
			self.attention[p2] = np.concatenate(attn_list, axis=0)
			self.latent[p2] = df_h_sc
			self.ylabel[p2] = df_h_sc.index.values

  
	def save(self):
		import h5py
		with h5py.File(self.wdir+'results/picasa_out.h5', 'w') as f:
			adata_keys = []
			for attr_name, attr_value in vars(self).items():
				if isinstance(attr_value, list) and '_keys' in attr_name:
					adata_keys = attr_value
			f.create_dataset('batch_keys', data=adata_keys,dtype=h5py.string_dtype(encoding='utf-8'))
			for k in adata_keys:
				f.create_dataset(k+'_latent', data=self.latent[k])
				f.create_dataset(k+'_attention', data=self.attention[k])
				f.create_dataset(k+'_ylabel', data=self.ylabel[k])

			for attr_name, attr_value in vars(self).items():
				if isinstance(attr_value, dict) and '_map' in attr_name:
					for k in attr_value.keys():
							df = pd.DataFrame(attr_value[k].items())
							f.create_dataset(k, data=df)

	def plot_loss(self):
		from picasa.util.plots import plot_loss
		for ad_pair in self.adata_pairs:
			p1 = self.adata_keys[ad_pair[0]]
			p2 = self.adata_keys[ad_pair[1]]

			plot_loss(self.wdir+'results/4_attncl_train_loss'+p1+'_'+p2+'.txt.gz',self.wdir+'results/4_attncl_train_loss'+p1+'_'+p2+'.png')

def create_picasa_object(adata_list: Adata,wdir: str):
	return picasa(dutil.data.Dataset(adata_list),wdir)
