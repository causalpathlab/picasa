import torch
torch.manual_seed(0)
import torch.nn as nn
from .loss import kl_loss,multi_dir_log_likelihood,reparameterize
import logging
logger = logging.getLogger(__name__)

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

class PICASAOUT:
	def __init__(self,z,theta,beta,tmean,tvar,bmean,bvar):
		self.z = z
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

		# x_sc = torch.log1p(x_sc)
		# x_sc = x_sc/torch.sum(x_sc,dim=-1,keepdim=True)
		
		row_sums = x_sc.sum(dim=1, keepdim=True)
		x_norm = torch.div(x_sc, row_sums) * 1e4


		h_sc = self.fc(x_norm)

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

class PICASAUNET(nn.Module):
	def __init__(self,input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches):
		super(PICASAUNET,self).__init__()
		
		self.encoder = Encoder(input_dim,unique_latent_dim,enc_layers)
		
		concat_dim = common_latent_dim + unique_latent_dim
		# concat_dim =  unique_latent_dim
		
		self.fusion_decoder = Stacklayers(concat_dim,dec_layers)

		decoder_in_dim = dec_layers[len(dec_layers)-1]

		self.decoder = Decoder(decoder_in_dim, input_dim)

	def forward(self,x_c1,x_zcommon):
		z_unique,theta_mean,theta_lnvar = self.encoder(x_c1)
		
		z_combined = torch.cat((x_zcommon, z_unique), dim=1)
  
		h = self.fusion_decoder(z_combined)

		beta_mean,beta_lnvar,theta,beta = self.decoder(h)
		
		return PICASAOUT(z_unique,theta,beta,theta_mean,theta_lnvar,beta_mean,beta_lnvar)
	
	

def train(model,data,l_rate,epochs):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate)
	epoch_losses = []
	data_size = len(data)
	for epoch in range(epochs):
		loss = 0
		loss_ll = 0
		loss_kl = 0
		loss_klb = 0
		for x_c1,y,x_zc,batch in data:
			opt.zero_grad()
			PICASAout = model(x_c1,x_zc)
			alpha = torch.exp(torch.clamp(torch.mm(PICASAout.theta,PICASAout.beta),-10,10))
			loglikloss = multi_dir_log_likelihood(x_c1,alpha)
			kl = kl_loss(PICASAout.theta_mean,PICASAout.theta_lnvar)
			klb = kl_loss(PICASAout.beta_mean,PICASAout.beta_lnvar)

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

		epoch_losses.append((loss_ll/len(data),loss_kl/len(data),loss_klb/len(data),0.0))

	return epoch_losses


def predict_batch(model,x_c1,y,x_zc):
	return model(x_c1,x_zc),y

