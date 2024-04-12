import torch
torch.manual_seed(0)
import torch.nn as nn
from .loss import kl_loss,multi_dir_log_likelihood,reparameterize
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
			self.input_size = next_l

	def forward(self, input_data):
		for layer in self.layers:
			input_data = layer(input_data)
		return input_data

	def get_activation(self):
		return nn.ReLU()


class SAILROUT:
	def __init__(self,z_sc,theta,beta,tmean,tvar,bmean,bvar):
		self.z_sc = z_sc
		self.theta = theta
		self.beta = beta

		self.theta_mean = tmean
		self.theta_lnvar = tvar

		self.beta_mean = bmean
		self.beta_lnvar = bvar
		
		
class Encoder(nn.Module):
	def __init__(self,input_dims,latent_dims,layers):
		super(Encoder, self).__init__()
		self.fc = Stacklayers(input_dims,layers)
		self.z_mean = nn.Linear(latent_dims,latent_dims)
		self.z_lnvar = nn.Linear(latent_dims,latent_dims)

	def forward(self, x_sc):

		x_sc = torch.log1p(x_sc)
		x_sc = x_sc/torch.sum(x_sc,dim=-1,keepdim=True)
		h_sc = self.fc(x_sc)

		theta_mean = self.z_mean(h_sc)
		theta_lnvar = torch.clamp(self.z_lnvar(h_sc),-4.0,4.0)
		z_sc = reparameterize(theta_mean,theta_lnvar)

		return z_sc, theta_mean, theta_lnvar


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
	def __init__(self,input_dims,latent_dims,encoder_layers):
		super(SAILRNET,self).__init__()
		self.encoder = Encoder(input_dims,latent_dims,encoder_layers)
		self.decoder = Decoder(latent_dims, input_dims)

	def forward(self,x_sc):
		z_sc,theta_mean,theta_lnvar = self.encoder(x_sc)
		beta_mean,beta_lnvar,theta,beta = self.decoder(z_sc)
		return SAILROUT(z_sc,theta,beta,theta_mean,theta_lnvar,beta_mean,beta_lnvar)

def train(model,data,epochs,l_rate):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate)
	loss_values = []
	loss_values_sep = []
	data_size = data.dataset.shape[0]
	for epoch in range(epochs):
		loss = 0
		loss_ll = 0
		loss_kl = 0
		loss_klb = 0
		for x_sc,y in data:
			opt.zero_grad()
			sailrout = model(x_sc)
			alpha = torch.exp(torch.clamp(torch.mm(sailrout.theta,sailrout.beta),-10,10))
			loglikloss = multi_dir_log_likelihood(x_sc,alpha)
			kl = kl_loss(sailrout.theta_mean,sailrout.theta_lnvar)
			klb = kl_loss(sailrout.beta_mean,sailrout.beta_lnvar)

			train_loss = torch.mean(kl -loglikloss).add(torch.sum(klb)/data_size)
			train_loss.backward()

			opt.step()

			ll_l = torch.mean(loglikloss).to('cpu')
			kl_l = torch.mean(kl).to('cpu')
			klb_l = torch.sum(klb).to('cpu')

			loss += train_loss.item()
			loss_ll += ll_l.item()
			loss_kl += kl_l.item()
			loss_klb += klb_l.item()/data_size

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))

		loss_values.append(loss/len(data))
		loss_values_sep.append((loss_ll/len(data),loss_kl/len(data),loss_klb/len(data)))


	return loss_values,loss_values_sep
def predict(model,data):
	for x_sc,y in data: break
	return model(x_sc),y
