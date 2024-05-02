import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from .loss import pcl_loss
import logging
logger = logging.getLogger(__name__)
from torch.distributions.uniform import Uniform
import pytorch_lightning as pl 
from pytorch_lightning.plugins import DDPPlugin
from torch.nn.parallel import DataParallel


class PICASAOUT:
	def __init__(self,h_sc,h_sp,z_sc,z_sp,attn_sc=None, attn_sp=None):
		self.h_sc = h_sc
		self.h_sp = h_sp
		self.z_sc = z_sc
		self.z_sp = z_sp
		self.attn_sc = attn_sc
		self.attn_sp = attn_sp

class PICASAel:
    def __init__(self,el_attn_sc,el_attn_sp, el_cl_sc,el_cl_sp,entropy_loss):
        self.el_attn_sc = el_attn_sc
        self.el_attn_sp = el_attn_sp
        self.el_cl_sc = el_cl_sc
        self.el_cl_sp = el_cl_sp
        self.entropy_loss = entropy_loss
        
           
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
  
	def forward(self, x):
		return self.fc(x)

class MLP(nn.Module):
	def __init__(self,input_dims,layers):
		super(MLP, self).__init__()
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x):
		z = self.fc(x)
		return z

class PICASANET(nn.Module):
	def __init__(self,input_dim, embedding_dim, attention_dim, latent_dim,encoder_layers,projection_layers,lambda_attention_sc_entropy_loss,lambda_attention_sp_entropy_loss,lambda_cl_sc_entropy_loss,lambda_cl_sp_entropy_loss):
		super(PICASANET,self).__init__()

		self.embedding = GeneEmbedor(embedding_dim,attention_dim)
		self.attention = ScaledDotAttention(attention_dim,input_dim)
		self.pooling = AttentionPooling(attention_dim)

		self.encoder = ENCODER(input_dim,encoder_layers)
		self.projector = MLP(latent_dim, projection_layers)
		
		self.lambda_attention_sc_entropy_loss = lambda_attention_sc_entropy_loss
		self.lambda_attention_sp_entropy_loss = lambda_attention_sp_entropy_loss
		self.lambda_cl_sc_entropy_loss = lambda_cl_sc_entropy_loss
		self.lambda_cl_sp_entropy_loss = lambda_cl_sp_entropy_loss

	def forward(self,x_sc,x_sp):
		x_sc_emb = self.embedding(x_sc)
		x_sp_emb = self.embedding(x_sp)
  
		x_sc_att_out, x_sc_att_w,el_attn_sc = self.attention(x_sc_emb,x_sp_emb,x_sp_emb)
		x_sc_pool_out = self.pooling(x_sc_att_out)

		x_sp_att_out, x_sp_att_w,el_attn_sp = self.attention(x_sp_emb,x_sc_emb,x_sc_emb)
		x_sp_pool_out = self.pooling(x_sp_att_out)

		h_sc = self.encoder(x_sc_pool_out)
		h_sp = self.encoder(x_sp_pool_out)

		z_sc = self.projector(h_sc)
		z_sp = self.projector(h_sp)
		
		pred_sc = torch.softmax(h_sc, dim=1)
		el_cl_sc = -torch.mean(torch.sum(pred_sc * torch.log(pred_sc + 1e-10), dim=1))

		pred_sp = torch.softmax(h_sp, dim=1)
		el_cl_sp = -torch.mean(torch.sum(pred_sp * torch.log(pred_sp + 1e-10), dim=1))

		entropy_loss = (el_attn_sc * self.lambda_attention_sc_entropy_loss +
						el_attn_sp * self.lambda_attention_sp_entropy_loss +
						el_cl_sc * self.lambda_cl_sc_entropy_loss +
						el_cl_sp * self.lambda_cl_sp_entropy_loss)

		return PICASAOUT(h_sc,h_sp,z_sc,z_sp,x_sc_att_w,x_sp_att_w), PICASAel(el_attn_sc,el_attn_sp, el_cl_sc,el_cl_sp,entropy_loss)

def train(model,data,epochs,l_rate,temperature,loss_file):
	logger.info('Init training....nn_attn')
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	for epoch in range(epochs):
		epoch_l, cl, el, el_attn_sc, el_attn_sp, el_cl_sc, el_cl_sp = (0,) * 7
		for x_sc,y,x_sp in data:
			opt.zero_grad()

			picasa_out,picasa_el = model(x_sc,x_sp)

			cl_loss = pcl_loss(picasa_out.z_sc, picasa_out.z_sp,  temperature)
			train_loss = cl_loss + picasa_el.entropy_loss
			train_loss.backward()

			opt.step()
			epoch_l += train_loss.item()
			cl += cl_loss.item()
			el += picasa_el.entropy_loss.item()
			el_attn_sc += picasa_el.el_attn_sc.item()
			el_attn_sp += picasa_el.el_attn_sp.item()
			el_cl_sc += picasa_el.el_cl_sc.item()
			el_cl_sp += picasa_el.el_cl_sp.item()
   		
		epoch_losses.append([epoch_l/len(data),cl/len(data),el/len(data),el_attn_sc/len(data),el_attn_sp/len(data),el_cl_sc/len(data),el_cl_sp/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))
	
	pd.DataFrame(epoch_losses,columns=['ep_l','cl','el','el_attn_sc','el_attn_sp','el_cl_sc','el_cl_sp']).to_csv(loss_file,index=False,compression='gzip')	
 

def predict(model,data):
	for x_sc,y, x_sp in data: break
	return model(x_sc,x_sp),y

def predict_batch(model,x_sc,y, x_sp ):
	return model(x_sc,x_sp),y

def predict_scsp(model,x_sc):
	x_sc_emb = model.embedding(x_sc)
	x_sp_emb = model.embedding(x_sc)

	x_sc_att_out, _,_ = model.attention(x_sc_emb,x_sp_emb,x_sp_emb)
	x_sc_pool_out = model.pooling(x_sc_att_out)

	x_sp_att_out, _,_ = model.attention(x_sp_emb,x_sc_emb,x_sc_emb)
	x_sp_pool_out = model.pooling(x_sp_att_out)

	h_sc = model.encoder(x_sc_pool_out)
	h_sp = model.encoder(x_sp_pool_out)

	z_sc = model.projector(h_sc)
	z_sp = model.projector(h_sp)
	
	return PICASAOUT(h_sc,h_sp,z_sc,z_sp)

class LitPICASANET(pl.LightningModule):
	
	def __init__(self,input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate,temperature,lossf):
		super(LitPICASANET,self).__init__()

		self.PICASAnet = PICASANET(input_dims, emb_dim, attn_dim, latent_dim,encoder_layers,projection_layers,features_low,features_high,corruption_rate)
		self.temperature = temperature 
		self.lossf = lossf
		self.PICASAnet = DataParallel(self.PICASAnet)
  
	def forward(self,x_sc,x_sp):
  
		return self.PICASAnet(x_sc,x_sp)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
		return optimizer

	def training_step(self,batch):

		x_sc,y,x_sp = batch
		device_of_model = next(self.PICASAnet.parameters()).device
		x_sc = x_sc.to(device_of_model)
		x_sp = x_sp.to(device_of_model)

		PICASAout = self.PICASAnet(x_sc,x_sp)

		train_loss = pcl_loss(PICASAout.z_sc, PICASAout.z_sp,  self.temperature)
	  
		f = open(self.lossf, 'a')
		f.write(str(torch.mean(train_loss).to('cpu').item()) + '\n')
		f.close()

