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
    def __init__(self,h_sc,h_spp,z_sc,z_spp,el):
        self.h_sc = h_sc
        self.h_spp = h_spp
        self.z_sc = z_sc
        self.z_spp = z_spp
        self.entropy_loss = el
        
class ENCODER(nn.Module):
	def __init__(self,input_dims,layers):
		super(ENCODER, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x):

		x = torch.log1p(x)
		x = x/torch.sum(x,dim=-1,keepdim=True)
		h = self.fc(x)
		return h

class MLP(nn.Module):
	def __init__(self,input_dims,layers):
		super(MLP, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, h):
		z = self.fc(h)
		return z


class SAILRNET(nn.Module):
	def __init__(self,input_dims,latent_dims,encoder_layers,projection_layers,entropy_weight):
		super(SAILRNET,self).__init__()
		self.encoder = ENCODER(input_dims,encoder_layers)
		self.projector = MLP(latent_dims, projection_layers)
		self.entroy_weight = entropy_weight

	def forward(self,x_sc, x_spp):
       
		h_sc = self.encoder(x_sc)
		h_spp = self.encoder(x_spp)

		z_sc = self.projector(h_sc)
		z_spp = self.projector(h_spp)
  
  		# entropy regularization
		pred_sc = torch.softmax(h_sc, dim=1)
		entropy_loss_sc = -torch.mean(torch.sum(pred_sc * torch.log(pred_sc + 1e-10), dim=1))

		pred_spp = torch.softmax(h_spp, dim=1)
		entropy_loss_spp = -torch.mean(torch.sum(pred_spp * torch.log(pred_spp + 1e-10), dim=1))

		entropy_loss = (entropy_loss_sc + entropy_loss_spp) * self.entroy_weight
  
		return SAILROUT(h_sc,h_spp,z_sc,z_spp,entropy_loss)

def train(model,data,epochs,l_rate,temperature):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	for epoch in range(epochs):
		loss = 0
		for x_sc,y,x_spp in data:
			opt.zero_grad()

			sailrout = model(x_sc,x_spp)

			train_loss = pcl_loss(sailrout.z_sc, sailrout.z_spp,temperature)	
			# train_loss = nt_xent_loss(sailrout.z_sc, sailrout.z_spp,temperature)	
			train_loss += sailrout.entropy_loss	
			train_loss.backward()

			opt.step()
			loss += train_loss.item()

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))


def predict(model,data):
	for x_sc,y, x_spp in data: break
	return model(x_sc,x_spp),y