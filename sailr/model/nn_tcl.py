import torch
torch.manual_seed(0)
import torch.nn as nn
from .loss import pcl_loss
import logging
logger = logging.getLogger(__name__)
from torch.distributions.uniform import Uniform

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


class SAILROUT:
	def __init__(self,h_sc,h_scc,z_sc,z_scc):
		self.h_sc = h_sc
		self.h_scc = h_scc
		self.z_sc = z_sc
		self.z_scc = z_scc
		
		
class ENCODER(nn.Module):
	def __init__(self,input_dims,layers):
		super(ENCODER, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x):

		x = torch.log1p(x)
		x = x/torch.sum(x,dim=-1,keepdim=True)
		z = self.fc(x)

		return z

class MLP(nn.Module):
	def __init__(self,input_dims,layers):
		super(MLP, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x):
		z = self.fc(x)
		return z


class SAILRNET(nn.Module):
	def __init__(self,input_dims,latent_dims,encoder_layers,projection_layers,features_low,features_high,corruption_rate):
		super(SAILRNET,self).__init__()
		self.encoder = ENCODER(input_dims,encoder_layers)
		self.projector = MLP(latent_dims, projection_layers)
		self.marginals = Uniform(features_low,features_high)
		self.corruption_rate = corruption_rate

	def forward(self,x_sc):
		corruption_mask = torch.randint_like(x_sc,high=x_sc.max()+1, device=x_sc.device) >  self.corruption_rate
		x_random = self.marginals.sample(torch.Size(x_sc.size())).to(x_sc.device)
		x_corrupted = torch.where(corruption_mask, x_random, x_sc)
  
		h_sc = self.encoder(x_sc)
		h_scc = self.encoder(x_corrupted)

		z_sc = self.projector(h_sc)
		z_scc = self.projector(h_scc)
  
		return SAILROUT(h_sc,h_scc,z_sc,z_scc)

def train(model,data,epochs,l_rate):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	for epoch in range(epochs):
		loss = 0
		for x_sc,y in data:
			opt.zero_grad()

			sailrout = model(x_sc)

			train_loss = pcl_loss(sailrout.z_sc, sailrout.z_scc)	
			train_loss.backward()

			opt.step()
			loss += train_loss.item()

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))


def predict(model,data):
	for x_sc,y in data: break
	return model(x_sc),y
