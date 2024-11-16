import torch
torch.manual_seed(0)
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from torch.distributions.uniform import Uniform
from .loss import get_zinb_reconstruction_loss, minimal_overlap_loss

		  
class Stacklayers(nn.Module):
	def __init__(self,input_size,layers,dropout=0.1):
		super(Stacklayers, self).__init__()
		self.layers = nn.ModuleList()
		self.input_size = input_size
		for next_l in layers:
			self.layers.append(nn.Linear(self.input_size,next_l))
			self.layers.append(nn.BatchNorm1d(next_l))
			self.layers.append(self.get_activation())
			self.layers.append(nn.Dropout(dropout))
			self.input_size = next_l

	def forward(self, input_data):
		for layer in self.layers:
			input_data = layer(input_data)
		return input_data

	def get_activation(self):
		return nn.ReLU()

class PICASAUNET(nn.Module):
	def __init__(self,input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches):
		super(PICASAUNET,self).__init__()
		self.u_encoder = Stacklayers(input_dim,enc_layers)

		concat_dim = common_latent_dim + unique_latent_dim 
		self.u_decoder = Stacklayers(concat_dim,dec_layers)
  
		decoder_in_dim = dec_layers[len(dec_layers)-1]
		self.zinb_scale = nn.Linear(decoder_in_dim, input_dim) 
		self.zinb_dropout = nn.Linear(decoder_in_dim, input_dim)
		self.zinb_dispersion = nn.Parameter(torch.randn(input_dim), requires_grad=True)
		
		self.batch_discriminator = nn.Linear(unique_latent_dim, num_batches)

	
	def forward(self,x_c1,x_zcommon):	
 
		row_sums = x_c1.sum(dim=1, keepdim=True)
		x_norm = torch.div(x_c1, row_sums) * 1e4
  
		z_unique = self.u_encoder(x_norm.float())
		
		z_combined = torch.cat((x_zcommon, z_unique), dim=1)

		h = self.u_decoder(z_combined)
  
		px_scale = torch.exp(self.zinb_scale(h))  
		px_dropout = self.zinb_dropout(h)  
		px_rate = self.zinb_dispersion.exp()
  
		batch_pred = self.batch_discriminator(z_unique)
		
		return z_unique,px_scale,px_rate,px_dropout,batch_pred
	

def train(model,data,l_rate,epochs=100):
	logger.info('Init training....nn_unq')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	for epoch in range(epochs):
		epoch_l,el_z,el_recon,el_batch = (0,)*4
		for x_c1,y,x_zc,batch in data:
			opt.zero_grad()
			z_u,px_s,px_r,px_d,batch_pred = model(x_c1,x_zc)
			train_loss_z = minimal_overlap_loss(x_zc,z_u)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch)
			train_loss = train_loss_z + train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_z += train_loss_z.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(data),el_z/len(data),el_recon/len(data),el_batch/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

	return epoch_losses

 
def predict_batch(model,x_c1,y,x_zc):
	return model(x_c1,x_zc),y
