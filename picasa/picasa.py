from .util.typehint import Adata
from . import dutil 
from . import model 
from . import neighbour
import torch
import logging
import gc
import os
import pandas as pd
import numpy as np
import itertools


class picasa(object):
	def __init__(self, data: dutil.data.Dataset, pair_mode:str,wdir: str):
		self.data = data
		self.wdir = wdir
		logging.basicConfig(filename=self.wdir+'results/4_attncl_train.log',
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S')
  
		self.adata_keys = list(self.data.adata_list.keys())

		if pair_mode == 'unq':
			indices = range(len(self.data.adata_list))
			adata_pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 1)]
			adata_pairs.append((adata_pairs[len(adata_pairs)-1][1],adata_pairs[0][0]))
			self.adata_pairs = adata_pairs
		else:
			indices = list(range(len(self.data.adata_list)))
			self.adata_pairs = list(itertools.combinations(indices, 2))
	
	def estimate_neighbour(self,method='approx_50'):
	 
		logging.info('Assign neighbour - '+ method)
		self.nbr_map = {}
  
		if 'approx' in method :

			number_of_trees = int(method.split('_')[1])
   
			for ad_pair in self.adata_pairs:
				p1 = self.adata_keys[ad_pair[0]]
				p2 = self.adata_keys[ad_pair[1]]
				logging.info('Generating neighbour using approximate method - ANNOY...\n'+p1+'_('+str(self.data.adata_list[p1].X.shape)+') >'+p2+'_('+str(self.data.adata_list[p2].X.shape)+')')
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
  
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		logging.info(picasa_model)
  
		loss = []

		for it in range(self.nn_params['titration']):
		
			logging.info('titration : '+ str(it))
  
			for ad_pair in self.adata_pairs:
				p1 = self.adata_keys[ad_pair[0]]
				p2 = self.adata_keys[ad_pair[1]]
	
				logging.info('Training...model-'+p1+'_'+p2)
		
				data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],self.nn_params['device'],self.nn_params['batch_size'])

				loss_p1_p2 = model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['lambda_loss'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'],self.nn_params['loss_clusters'],self.nn_params['loss_threshold'], self.nn_params['loss_weight'],self.nn_params['temperature_cl'],self.nn_params['loss_clusters'])
	
				logging.info('Training...model-'+p2+'_'+p1)
	
				data = dutil.nn_load_data_pairs(self.data.adata_list[p2],self.data.adata_list[p1],self.nbr_map[p2+'_'+p1],self.nn_params['device'],self.nn_params['batch_size'])

				loss_p2_p1 = model.nn_attn.train(picasa_model,data,self.nn_params['epochs'],self.nn_params['lambda_loss'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'],self.nn_params['loss_clusters'],self.nn_params['loss_threshold'], self.nn_params['loss_weight'],self.nn_params['temperature_cl'],self.nn_params['loss_clusters'])

				loss_p1_p2 = np.array(loss_p1_p2)
				loss_p2_p1 = np.array(loss_p2_p1)
				stacked_loss_p = np.vstack((loss_p1_p2, loss_p2_p1))
				loss.append(np.mean(stacked_loss_p, axis=0))

		torch.save(picasa_model.state_dict(),self.wdir+'results/nn_attncl.model')
		pd.DataFrame(loss,columns=['ep_l','cl','el','el_attn_sc','el_attn_sp','el_cl_sc','el_cl_sp']).to_csv(self.wdir+'results/4_attncl_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in results/nn_attncl.model')
  
	def eval_model(self,eval_batch_size,eval_total_size,device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
  
		picasa_model.eval()
	
		self.latent = {}
		self.ylabel = {}

		evaled = []
  
		for ad_pair in list(self.nbr_map.keys()):
			
			if len(ad_pair.split('_')) == 2:
				p1 = ad_pair.split('_')[0]
				p2 = ad_pair.split('_')[1]
			else:
				p1 = ad_pair.split('_')[0]+'_'+ad_pair.split('_')[1]
				p2 = ad_pair.split('_')[2]+'_'+ad_pair.split('_')[3]

			if p1 not in evaled:
				logging.info('eval :'+p1+'_'+p2)
				data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],device,eval_batch_size)
		
				df_latent = pd.DataFrame()

				for x_c1,y,x_c2,nbr_weight in data_pred:
					m_el,ylabel = model.nn_attn.predict_batch(picasa_model,x_c1,y,x_c2)
					m = m_el[0]
					df_latent = pd.concat([df_latent,pd.DataFrame(m.h_c1.cpu().detach().numpy(),index=ylabel)],axis=0)

					del x_c1, y, x_c2, m_el, ylabel
					gc.collect()

					if df_latent.shape[0]>eval_total_size:
						break

		
				self.latent[p1] = df_latent
				self.ylabel[p1] = df_latent.index.values
				evaled.append(p1)

   
	def eval_attention(self,adata_p1, adata_p2,adata_nbr_map,eval_batch_size,eval_total_size,device='cpu'):
		
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
		picasa_model.eval()
  
		data_pred = dutil.nn_load_data_pairs(adata_p1, adata_p2, adata_nbr_map,device,eval_batch_size)

		attn_list = []
		ylabel_list = []
		y_count = 0

		for x_c1,y,x_c2,nbr_weight in data_pred:
			x_c1_attn = model.nn_attn.predict_attention(picasa_model,x_c1,x_c2)
			attn_list.append(x_c1_attn.cpu().detach().numpy())
			ylabel_list.append(y)
			
			y_count += len(y)
			if y_count>eval_total_size:
				break
			
			del x_c1_attn, y
			gc.collect()

		attn_list = np.concatenate(attn_list, axis=0)
		ylabel_list = np.concatenate(ylabel_list, axis=0)

		return attn_list,ylabel_list


		# attn_file_p1 = self.wdir+'results/'+p1+'_attention.npz'
		# index_file_p1 = self.wdir+'results/'+p1+'_index.csv.gz'
			
		# np.savez_compressed(attn_file_p1, np.concatenate(attn_list, axis=0))
		# pd.DataFrame(ylabel_list).index.to_series().to_csv(index_file_p1, compression='gzip')


	def eval_context(self,adata_p1, adata_p2,adata_nbr_map,eval_batch_size,eval_total_size, device='cpu'):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
		picasa_model.eval()
  
		data_pred = dutil.nn_load_data_pairs(adata_p1, adata_p2, adata_nbr_map,device,eval_batch_size)

		context_list = []
		ylabel_list = []
		y_count = 0
  
		for x_c1,y,x_c2,nbr_weight in data_pred:
			x_context = model.nn_attn.predict_context(picasa_model,x_c1,x_c2)
			context_list.append(x_context.cpu().detach().numpy())
			ylabel_list.append(y)

			y_count += len(y)
			if y_count>eval_total_size:
				break

			del x_context, y
			gc.collect()


		context_list = np.concatenate(context_list, axis=0)
		ylabel_list = np.concatenate(ylabel_list, axis=0)

		return context_list,ylabel_list
  

	def train_unique(self,unq_layers,l_rate,epochs,batch_size,device):
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(self.nn_params['device'])))
		picasa_model.eval()

		picasa_unq_model = model.nn_unq.PICASAUNET(picasa_model,self.nn_params['latent_dim'],self.nn_params['input_dim'],unq_layers).to(self.nn_params['device'])
	
		logging.info(picasa_unq_model)
  
		x_c1_batches = []
		y_batches = []
		x_c2_batches = []
		b_ids_batches = []
  
		logging.info("Creating dataloader for training unique space.")
  
		for ad_pair in self.adata_pairs:
      
			p1 = self.adata_keys[ad_pair[0]]
			p2 = self.adata_keys[ad_pair[1]]
			
			data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],'cpu',batch_size)
	
			for x_c1,y,x_c2,nbr_weight in data:		
				x_c1_batches.append(x_c1)
				y_batches.append(np.array(y))
				x_c2_batches.append(x_c2)
				b_ids_batches.append(np.array([p1+'+'+p2 for x in range(x_c1.shape[0])]))
	
		all_x_c1 = torch.cat(x_c1_batches, dim=0)  
		all_y = np.concatenate(y_batches)        
		all_x_c2 = torch.cat(x_c2_batches, dim=0)  
		all_b_ids = np.concatenate(b_ids_batches)  

		train_data =dutil.get_dataloader_mem(all_x_c1,all_y,all_x_c2,all_b_ids,batch_size,device)

		logging.info('Training...unique space model-'+p1+'_'+p2)
		loss = []
		loss_p1_p2 = model.nn_unq.train(picasa_unq_model,train_data,l_rate,epochs)
		loss.append(np.mean(loss_p1_p2, axis=0))

		torch.save(picasa_unq_model.state_dict(),self.wdir+'results/nn_unq.model')
		pd.DataFrame(loss,columns=['ep_l']).to_csv(self.wdir+'results/4_unq_train_loss.txt.gz',index=False,compression='gzip',header=True)
		logging.info('Completed training...model saved in results/nn_unq.model')
  
  

	def eval_unique(self,unq_layers,eval_batch_size, eval_total_size,device):
	 
		picasa_model = model.nn_attn.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
		picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
  
		picasa_model.eval()
	
	
		picasa_unq_model = model.nn_unq.PICASAUNET(picasa_model,self.nn_params['latent_dim'],self.nn_params['input_dim'],unq_layers).to(self.nn_params['device'])

		picasa_unq_model.load_state_dict(torch.load(self.wdir+'results/nn_unq.model', map_location=torch.device(device)))

		picasa_unq_model.eval()
  
		c_latent = {}
		u_latent = {}
  
		for ad_pair in self.adata_pairs:
			p1 = self.adata_keys[ad_pair[0]]
			p2 = self.adata_keys[ad_pair[1]]

			logging.info('Training...model-'+p1+'_'+p2)
	
			data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],self.nn_params['device'],self.nn_params['batch_size'])

  
			
			df_c_latent = pd.DataFrame()
			df_u_latent = pd.DataFrame()

			for x_c1,y,x_c2,nbr_weight in data:
				z,ylabel = model.nn_unq.predict_batch(picasa_unq_model,x_c1,y,x_c2)
				z_c = z[0]
				z_u = z[1]
				df_c_latent = pd.concat([df_c_latent,pd.DataFrame(z_c.cpu().detach().numpy(),index=ylabel)],axis=0)
				df_u_latent = pd.concat([df_u_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel)],axis=0)

				del z_c, z_u, ylabel
				gc.collect()

				if df_c_latent.shape[0]>eval_total_size:
					break
			
			c_latent[p1+'_common'] = df_c_latent			
			u_latent[p1+'_unique'] = df_u_latent			
		
		return c_latent,u_latent
   


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
				f.create_dataset(k+'_ylabel', data=self.ylabel[k])

			for attr_name, attr_value in vars(self).items():
				if isinstance(attr_value, dict) and '_map' in attr_name:
					for k in attr_value.keys():
							df = pd.DataFrame(attr_value[k].items())
							df = df[['target', 'weight']] = pd.DataFrame(df[1].tolist(), index=df.index.values)
							df = df.drop(columns=[0,1])
							df.reset_index(inplace=True)
							df.columns = ['source','target','weight']
							f.create_dataset(k, data=df)

	def plot_loss(self,tag):
		from picasa.util.plots import plot_loss
		if tag=='common':
			plot_loss(self.wdir+'results/4_attncl_train_loss.txt.gz',self.wdir+'results/4_attncl_train_loss.png')
		elif tag=='unq':
			plot_loss(self.wdir+'results/4_unq_train_loss.txt.gz',self.wdir+'results/4_unq_attncl_train_loss.png')

def create_picasa_object(adata_list: Adata, pair_mode: str,wdir: str):
	return picasa(dutil.data.Dataset(adata_list),pair_mode,wdir)
