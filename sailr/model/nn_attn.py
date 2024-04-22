import torch
torch.manual_seed(0)
import torch.nn as nn
from .loss import pcl_loss
import logging
logger = logging.getLogger(__name__)
from torch.distributions.uniform import Uniform
import pytorch_lightning as pl 
from pytorch_lightning.plugins import DDPPlugin

class SAILROUT:
	def __init__(self,h_sc,h_spp,z_sc,z_spp,attn_sc, attn_spp):
		self.h_sc = h_sc
		self.h_spp = h_spp
		self.z_sc = z_sc
		self.z_spp = z_spp
		self.attn_sc = attn_sc
		self.attn_spp = attn_spp

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

class GeneEmbedor(nn.Module):
	def __init__(self,emb_dim,out_dim):
		super(GeneEmbedor, self).__init__()
		self.embedding = nn.Embedding(emb_dim,out_dim)
		self.emb_norm = nn.LayerNorm(out_dim)

	def forward(self, x):
		x = self.embedding(x)
		x = self.emb_norm(x)
		return x

class ScaledDotAttention(nn.Module):
	
	def __init__(self, weight_dim):
		super(ScaledDotAttention, self).__init__()
		self.W_query = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.W_key = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.W_value = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.model_dim = weight_dim
		
	def forward(self, query, key, value):

		query_proj = torch.matmul(query, self.W_query)
		key_proj = torch.matmul(key, self.W_key)
		value_proj = torch.matmul(value, self.W_value)
		
		scores = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.model_dim).float())
		attention_weights = torch.softmax(scores, dim=-1)
		output = torch.matmul(attention_weights, value_proj)
		
		return output, attention_weights

class AttentionPooling(nn.Module):
	def __init__(self, model_dim):
		super(AttentionPooling, self).__init__()
		self.weights = nn.Parameter(torch.randn(model_dim))  
	
	def forward(self, attention_output):
		weights_softmax = torch.softmax(self.weights, dim=0)
		weighted_output = attention_output * weights_softmax.unsqueeze(0)
		pooled_output = torch.sum(weighted_output, dim=-1, keepdim=True)
		return pooled_output.squeeze(-1)

class ENCODER(nn.Module):
	def __init__(self,input_dims,layers):
		super(ENCODER, self).__init__()
		self.fc = Stacklayers(input_dims,layers)
		self.depth = 1e4
  
	def forward(self, x):

		# x = torch.log1p(x)
		# x = x/torch.sum(x,dim=-1,keepdim=True)
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
	def __init__(self,input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate):
		super(SAILRNET,self).__init__()

		self.embedding = GeneEmbedor(emb_dim,attn_dim)
		self.attention = ScaledDotAttention(attn_dim)
		self.pooling = AttentionPooling(attn_dim)

		self.encoder = ENCODER(input_dims,encoder_layers)
		self.projector = MLP(latent_dim, projection_layers)

	def forward(self,x_sc,x_spp):
		x_sc_emb = self.embedding(x_sc)
		x_spp_emb = self.embedding(x_spp)
  
		x_sc_att_out, x_sc_att_w = self.attention(x_sc_emb,x_spp_emb,x_sc_emb)
		x_sc_pool_out = self.pooling(x_sc_att_out)

		x_spp_att_out, x_spp_att_w = self.attention(x_spp_emb,x_sc_emb,x_spp_emb)
		x_spp_pool_out = self.pooling(x_spp_att_out)

		h_sc = self.encoder(x_sc_pool_out)
		h_spp = self.encoder(x_spp_pool_out)

		z_sc = self.projector(h_sc)
		z_spp = self.projector(h_spp)
  
		return SAILROUT(h_sc,h_spp,z_sc,z_spp,x_sc_att_w,x_spp_att_w)

def train(model,data,epochs,l_rate,temperature):
	logger.info('Starting training....')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	for epoch in range(epochs):
		loss = 0
		for x_sc,y,x_spp in data:
			opt.zero_grad()

			sailrout = model(x_sc,x_spp)

			train_loss = pcl_loss(sailrout.z_sc, sailrout.z_spp,  temperature)	
			train_loss.backward()

			opt.step()
			loss += train_loss.item()

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))

def predict(model,data):
	for x_sc,y, x_spp in data: break
	return model(x_sc,x_spp),y


class LitSAILRNET(pl.LightningModule):
	def __init__(self,input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate,temperature,lossf):
		super(LitSAILRNET,self).__init__()

		self.sailrnet = SAILRNET(input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate)
		self.temperature = temperature 
		self.lossf = lossf
  
	def forward(self,x_sc,x_spp):
  
		return self.sailrnet(x_sc,x_spp)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
		return optimizer

	def training_step(self,batch):

		x_sc,y,x_spp = batch

		sailrout = self.sailrnet(x_sc,x_spp)

		train_loss = pcl_loss(sailrout.z_sc, sailrout.z_spp,  self.temperature)
  	
		f = open(self.lossf, 'a')
		f.write(str(torch.mean(train_loss).to('cpu').item()) + '\n')
		f.close()

