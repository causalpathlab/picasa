import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import pcl_loss,latent_alignment_loss, minimal_overlap_loss, get_zinb_reconstruction_loss
				
import logging
logger = logging.getLogger(__name__)
import numpy as np
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def picasa_train_common(model,data,
	epochs:int,
	l_rate:float,
	cl_loss_mode:str, 
	min_batchsize:int = 5
	):
	
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	for epoch in range(epochs):
		epoch_l, cl, al = (0,) * 3
		for x_c1,y,x_c2,nbr_weight in data:
						
			if x_c1.shape[0] < min_batchsize:
				continue
			
			opt.zero_grad()

			picasa_out = model(x_c1,x_c2,nbr_weight)

			cl_loss = pcl_loss(picasa_out.z_c1, picasa_out.z_c2,cl_loss_mode)
			
			alignment_loss = latent_alignment_loss(picasa_out.h_x1, picasa_out.h_x2)

			train_loss = cl_loss + alignment_loss
			
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			cl += cl_loss.item() 
			al +=  alignment_loss.item()
		   
		epoch_losses.append([epoch_l/len(data),cl/len(data),al/len(data),0.0])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f} , {:.4f} , {:.4f}'.format(epoch+1,epoch_l/len(data),cl/len(data),al/len(data) ))

		return epoch_losses


def picasa_train_unique(model,data,l_rate,epochs=100):

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
 

def picasa_train_base(model,data,l_rate,epochs=100):

	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	criterion = nn.CrossEntropyLoss() 
	for epoch in range(epochs):
		epoch_l,el_recon,el_batch = (0,)*3
		for x_c1,y,batch in data:
			opt.zero_grad()
			z,px_s,px_r,px_d,batch_pred = model(x_c1)
			train_loss_recon = get_zinb_reconstruction_loss(x_c1,px_s, px_r, px_d)
			train_loss_batch = criterion(batch_pred, batch)
			train_loss = train_loss_recon + train_loss_batch
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			el_recon += train_loss_recon.item()
			el_batch += train_loss_batch.item()
		   
		epoch_losses.append([epoch_l/len(data),0.0,el_recon/len(data),el_batch/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

	return epoch_losses
 