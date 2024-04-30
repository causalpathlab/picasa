import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from .loss import pcl_loss
import logging
logger = logging.getLogger(__name__)
from torch.distributions.uniform import Uniform
import pytorch_lightning as pl 
from pytorch_lightning.plugins import DDPPlugin
from torch.nn.parallel import DataParallel
import math

class SAILROUT:
	def __init__(self,h_sc,h_spp,z_sc,z_spp,attn_sc=None, attn_spp=None, el=None):
		self.h_sc = h_sc
		self.h_spp = h_spp
		self.z_sc = z_sc
		self.z_spp = z_spp
		self.attn_sc = attn_sc
		self.attn_spp = attn_spp
		self.entropy_loss = el

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

class PosEmbedor(nn.Module):
	def __init__(self,emb_dim,out_dim):
		super(PosEmbedor, self).__init__()
		self.embedding = nn.Embedding(emb_dim,out_dim)
		self.emb_norm = nn.LayerNorm(out_dim)

	def forward(self, x):
		x = self.embedding(x)
		x = self.emb_norm(x)
		return x


class ScaledDotAttention(nn.Module):
	
	def __init__(self, weight_dim,input_dim):
		super(ScaledDotAttention, self).__init__()
		self.W_query = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.W_key = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.W_value = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.model_dim = weight_dim
		self.self_importance = nn.Parameter(torch.zeros(input_dim))
		
	def forward(self, query, key, value):

		query_proj = torch.matmul(query, self.W_query)
		key_proj = torch.matmul(key, self.W_key)
		value_proj = torch.matmul(value, self.W_value)
		

		scores = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.model_dim).float())
		diag_bias = torch.eye(scores.shape[1], dtype=scores.dtype, device=scores.device)
		importance = torch.clamp(torch.exp(self.self_importance),max=3)
		scores += (diag_bias * importance)
  
		attention_weights = torch.softmax(scores, dim=-1)
		entropy_loss_attn = -torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1))

		output = torch.matmul(attention_weights, value_proj)
		
		return output, attention_weights,entropy_loss_attn

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
	def __init__(self,input_dims, emb_dim, pos_emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers):
		super(SAILRNET,self).__init__()

		self.embedding = GeneEmbedor(emb_dim,attn_dim)
		self.pos_embedding = PosEmbedor(pos_emb_dim,attn_dim)
		self.attention = ScaledDotAttention(attn_dim,input_dims)
		self.pooling = AttentionPooling(attn_dim)

		self.encoder = ENCODER(input_dims,encoder_layers)
		self.projector = MLP(latent_dim, projection_layers)
		
		self.entropy_loss_cl = 0.1
		self.entropy_loss_attn = 1.0
  
	def forward(self,x_sc,x_spp,sp_x,sp_y):
		x_sc_emb = self.embedding(x_sc)
		x_spp_emb = self.embedding(x_spp)

		x_spp_x_emb = self.pos_embedding(sp_x)
		x_spp_y_emb = self.pos_embedding(sp_y)

		input_d = x_spp_emb.shape[1]
		batch_d = x_spp_x_emb.shape[0]
		attn_d = x_spp_x_emb.shape[1]

		x_spp_x_emb = x_spp_x_emb.unsqueeze(1).expand(batch_d, input_d, attn_d)
		x_spp_y_emb = x_spp_y_emb.unsqueeze(1).expand(batch_d, input_d, attn_d)
		x_spp_emb += (x_spp_x_emb+x_spp_y_emb) 
  
		x_sc_att_out, x_sc_att_w,el_attn = self.attention(x_sc_emb,x_spp_emb,x_spp_emb)
		x_sc_pool_out = self.pooling(x_sc_att_out)

		x_spp_att_out, x_spp_att_w,el_attn = self.attention(x_spp_emb,x_sc_emb,x_sc_emb)
		x_spp_pool_out = self.pooling(x_spp_att_out)

		h_sc = self.encoder(x_sc_pool_out)
		h_spp = self.encoder(x_spp_pool_out)

		z_sc = self.projector(h_sc)
		z_spp = self.projector(h_spp)
		
		pred_sc = torch.softmax(h_sc, dim=1)
		entropy_loss_sc = -torch.mean(torch.sum(pred_sc * torch.log(pred_sc + 1e-10), dim=1))

		pred_spp = torch.softmax(h_spp, dim=1)
		entropy_loss_spp = -torch.mean(torch.sum(pred_spp * torch.log(pred_spp + 1e-10), dim=1))

		entropy_loss = (entropy_loss_sc + entropy_loss_spp) * self.entropy_loss_cl
  
		entropy_loss += (el_attn) * self.entropy_loss_attn
  
		return SAILROUT(h_sc,h_spp,z_sc,z_spp,x_sc_att_w,x_spp_att_w,entropy_loss)

def train(model,data,epochs,l_rate,temperature):
	logger.info('Starting training....nn_attn')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	for epoch in range(epochs):
		loss = 0
		for x_sc,y,x_spp,sp_x,sp_y in data:
			opt.zero_grad()

			sailrout = model(x_sc,x_spp,sp_x,sp_y)

			train_loss = pcl_loss(sailrout.z_sc, sailrout.z_spp,  temperature)
			train_loss += sailrout.entropy_loss
			train_loss.backward()

			opt.step()
			loss += train_loss.item()

		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss/len(data)))

def predict(model,data):
	for x_sc,y, x_spp,sp_x,sp_y in data: break
	return model(x_sc,x_spp,sp_x,sp_y),y

def predict_batch(model,x_sc,y, x_spp ):
	return model(x_sc,x_spp),y

def predict_scsp(model,x_sc):
	x_sc_emb = model.embedding(x_sc)
	x_spp_emb = model.embedding(x_sc)

	x_sc_att_out, _,_ = model.attention(x_sc_emb,x_spp_emb,x_spp_emb)
	x_sc_pool_out = model.pooling(x_sc_att_out)

	x_spp_att_out, _,_ = model.attention(x_spp_emb,x_sc_emb,x_sc_emb)
	x_spp_pool_out = model.pooling(x_spp_att_out)

	h_sc = model.encoder(x_sc_pool_out)
	h_spp = model.encoder(x_spp_pool_out)

	z_sc = model.projector(h_sc)
	z_spp = model.projector(h_spp)
	
	return SAILROUT(h_sc,h_spp,z_sc,z_spp)

class LitSAILRNET(pl.LightningModule):
	
	def __init__(self,input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate,temperature,lossf):
		super(LitSAILRNET,self).__init__()

		self.sailrnet = SAILRNET(input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate)
		self.temperature = temperature 
		self.lossf = lossf
		self.sailrnet = DataParallel(self.sailrnet)
  
	def forward(self,x_sc,x_spp):
  
		return self.sailrnet(x_sc,x_spp)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
		return optimizer

	def training_step(self,batch):

		x_sc,y,x_spp = batch
		device_of_model = next(self.sailrnet.parameters()).device
		x_sc = x_sc.to(device_of_model)
		x_spp = x_spp.to(device_of_model)

		sailrout = self.sailrnet(x_sc,x_spp)

		train_loss = pcl_loss(sailrout.z_sc, sailrout.z_spp,  self.temperature)
	  
		f = open(self.lossf, 'a')
		f.write(str(torch.mean(train_loss).to('cpu').item()) + '\n')
		f.close()

