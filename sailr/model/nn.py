import torch
torch.manual_seed(0)
import torch.nn as nn
from .loss import kl_loss,multi_dir_log_likelihood,reparameterize,triplet_loss
import logging
logger = logging.getLogger(__name__)

class Stacklayers(nn.Module):
	def __init__(self,input_size,layers):
		super(Stacklayers, self).__init__()
		self.layers = nn.ModuleList()
		self.input_size = input_size
		for next_l in layers:
			self.layers.append(nn.Linear(self.input_size,next_l))
			self.layers.append(self.get_activation())
			nn.BatchNorm1d(next_l)
			self.input_size = next_l

	def forward(self, input_data):
		for layer in self.layers:
			input_data = layer(input_data)
		return input_data

	def get_activation(self):
		return nn.ReLU()

class ProjectionHead(nn.Module):
	def __init__(self,latent_dims,layers):
		super(ProjectionHead, self).__init__()
		self.fc = Stacklayers(latent_dims,layers)
  
	def forward(self, xx):
		return self.fc(xx)

class SAILROUT:
    def __init__(self,h_sc,h_spp,h_spn,z_sc,z_spp,z_spn,theta,beta,bmean,bvar):
        self.h_sc = h_sc
        self.h_spp = h_spp
        self.h_spn = h_spn
        self.z_sc = z_sc
        self.z_spp = z_spp
        self.z_spn = z_spn
        self.theta = theta
        self.beta = beta
        self.bmean = bmean
        self.bvar = bvar
        
        
class Encoder(nn.Module):
	def __init__(self,input_dims,layers):
		super(Encoder, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x_sc, x_spp, x_spn):

		x_sc = torch.log1p(x_sc)
		x_sc = x_sc/torch.sum(x_sc,dim=-1,keepdim=True)
		h_sc = self.fc(x_sc)

		x_spp = torch.log1p(x_spp)
		x_spp = x_spp/torch.sum(x_spp,dim=-1,keepdim=True)
		h_spp = self.fc(x_spp)

		x_spn = torch.log1p(x_spn)
		x_spn = x_spn/torch.sum(x_spn,dim=-1,keepdim=True)
		h_spn = self.fc(x_spn)

		return h_sc,h_spp,h_spn


class Decoder(nn.Module):
	def __init__(self,latent_dims,out_dims,jitter=.1):
		super(Decoder, self).__init__()
		
		self.beta_bias= nn.Parameter(torch.randn(1,out_dims)*jitter)
		self.beta_mean = nn.Parameter(torch.randn(latent_dims,out_dims)*jitter)
		self.beta_lnvar = nn.Parameter(torch.zeros(latent_dims,out_dims))

		self.lsmax = nn.LogSoftmax(dim=-1)

	def forward(self, zz):
		
		theta = torch.exp(self.lsmax(zz))
		
		z_beta = self.get_beta()
		beta = z_beta.add(self.beta_bias)

		return self.beta_mean,self.beta_lnvar,theta,beta

	def get_beta(self):
		lv = torch.clamp(self.beta_lnvar,-5.0,5.0)
		z_beta = reparameterize(self.beta_mean,lv) 
		return z_beta

class SAILRNET(nn.Module):
	def __init__(self,input_dims,latent_dims,layers):
		super(SAILRNET,self).__init__()
		self.encoder = Encoder(input_dims,layers)
		self.projector = ProjectionHead(latent_dims,layers)
		self.decoder = Decoder(latent_dims, input_dims)

	def forward(self,x_sc, x_spp, x_spn):
		h_sc,h_spp,h_spn = self.encoder(x_sc,x_spp,x_spn)
		z_sc = self.projector(h_sc)
		z_spp = self.projector(h_spp)
		z_spn = self.projector(h_spn)
		bm,bv,theta,beta = self.decoder(h_sc)
		return SAILROUT(h_sc,h_spp,h_spn,z_sc,z_spp,z_spn,theta,beta,bm,bv)

def train(model,data,epochs,l_rate,batch_size):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate)
	loss_values = []
	loss_values_sep = []

	for epoch in range(epochs):
		loss = 0
		loss_ll = 0
		loss_kl = 0
		loss_cl = 0
		for x_sc,y,x_spp,x_spn in data:
      
			opt.zero_grad()
   
			sailrout = model(x_sc,x_spp,x_spn)
   
			alpha = torch.exp(torch.clamp(torch.mm(sailrout.theta,sailrout.beta),-10,10))
			loglikloss = multi_dir_log_likelihood(x_sc,alpha).mean()/10
   
			klb = kl_loss(sailrout.bmean,sailrout.bvar).sum()

			cl = triplet_loss(sailrout.z_sc,sailrout.z_spp,sailrout.z_spn)
   

			train_loss = loglikloss + klb + cl
   
			train_loss.backward()
   
			opt.step()

			ll_l = loglikloss.to('cpu')
			kl_l = klb.to('cpu')
			cl_l = cl.to('cpu')

			loss += train_loss.item()
			loss_ll += ll_l.item()
			loss_kl += kl_l.item()
			loss_cl += cl_l.item()

		# if epoch % 10 == 0:
		print('====> Epoch: {} Average loss: {:.4f},{:.4f},{:.4f},{:.4f}'.format(epoch, loss/batch_size,loss_ll/batch_size,loss_kl/batch_size,loss_cl/batch_size))

		# loss_values.append(loss/len(data))
		# loss_values_sep.append((loss_ll/len(data),loss_kl/len(data)))


	return loss_values,loss_values_sep