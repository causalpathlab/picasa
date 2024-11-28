import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import pcl_loss,pcl_loss_with_rare_cluster,pcl_loss_with_weighted_cluster,\
                latent_alignment_loss, minimal_overlap_loss, get_zinb_reconstruction_loss
import logging
logger = logging.getLogger(__name__)
import numpy as np
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def picasa_train_common(model,data,
    epochs:int,
    lambda_loss:float,
    l_rate:float,
    cl_loss_mode:str, 
    loss_clusters:float, 
    loss_threshold:float, 
    loss_weight:float,
    temperature:float,
    min_batchsize:int
    ):
    
    opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
    epoch_losses = []
    lambda_attn_loss = float(lambda_loss[0])
    lambda_latent_loss = float(lambda_loss[1])
    lambda_latalign_loss = float(lambda_loss[2])
    lambda_cl_loss = float(lambda_loss[3])
    for epoch in range(epochs):
        epoch_l, cl, el, el_attn_c1, el_attn_c2, el_cl_c1, el_cl_c2 = (0,) * 7
        for x_c1,y,x_c2,nbr_weight in data:
                        
            if x_c1.shape[0] < min_batchsize:
                continue
            
            opt.zero_grad()

            picasa_out,picasa_el = model(x_c1,x_c2,nbr_weight)

            if cl_loss_mode == 'rare':
                cl_loss = lambda_cl_loss * pcl_loss_with_rare_cluster(picasa_out.z_c1, picasa_out.z_c2,loss_clusters, loss_threshold, loss_weight,temperature)
            elif cl_loss_mode == 'weighted':
                cl_loss = lambda_cl_loss * pcl_loss_with_weighted_cluster(picasa_out.z_c1, picasa_out.z_c2,loss_clusters,loss_weight,temperature)
            else:
                cl_loss = lambda_cl_loss * pcl_loss(picasa_out.z_c1, picasa_out.z_c2,temperature)


            entropy_loss = (picasa_el.el_attn_c1 * lambda_attn_loss +
                        picasa_el.el_attn_c2 * lambda_attn_loss +
                        picasa_el.el_cl_c1 * lambda_latent_loss +
                        picasa_el.el_cl_c2 * lambda_latent_loss)
            
            alignment_loss = latent_alignment_loss(picasa_out.z_c1, picasa_out.z_c2) * lambda_latalign_loss

            train_loss = cl_loss + entropy_loss + alignment_loss
            
            train_loss.backward()

            opt.step()
            epoch_l += train_loss.item()
            cl += cl_loss.item() + alignment_loss.item()
            el += entropy_loss.item()
            el_attn_c1 += picasa_el.el_attn_c1.item()
            el_attn_c2 += picasa_el.el_attn_c2.item()
            el_cl_c1 += picasa_el.el_cl_c1.item()
            el_cl_c2 += picasa_el.el_cl_c2.item()
           
        epoch_losses.append([epoch_l/len(data),cl/len(data),el/len(data),el_attn_c1/len(data),el_attn_c2/len(data),el_cl_c1/len(data),el_cl_c2/len(data)])  
        
        if epoch % 10 == 0:
            logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

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
 