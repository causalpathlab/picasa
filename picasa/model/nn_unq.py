import torch
torch.manual_seed(0)
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
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

class PICASAUNET(nn.Module):
    def __init__(self,picasa_model,latent_dim,layers ):
        super(PICASAUNET,self).__init__()
        self.p_model = picasa_model
        self.u_encoder = Stacklayers(latent_dim,layers)
        
        
        for param in self.p_model.parameters():
            param.requires_grad = False
    
    def forward(self,x_c1,x_c2):
        x_c1_emb = self.p_model.embedding(x_c1)
        x_c2_emb = self.p_model.embedding(x_c2)
        x_c1_context,_,_ = self.p_model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
        x_c1_pool_out = self.p_model.pooling(x_c1_context)
        z_common = self.p_model.encoder(x_c1_pool_out)
    
        z_unique = self.u_encoder(z_common)
        return z_common,z_unique
    
def minimal_overlap_loss(z_common, z_unique):
    z_common_norm = F.normalize(z_common, p=2, dim=-1)
    z_unique_norm = F.normalize(z_unique, p=2, dim=-1)
    
    cosine_similarity = torch.sum(z_common_norm * z_unique_norm, dim=-1)
    return torch.mean(cosine_similarity)


def train(model,data,l_rate,epochs=100):
    logger.info('Init training....nn_unq')
    opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
    epoch_losses = []
    for epoch in range(epochs):
        print(epoch)
        epoch_l = 0
        for x_c1,y,x_c2 in data:
            opt.zero_grad()

            z_c,z_u = model(x_c1,x_c2)
            train_loss = minimal_overlap_loss(z_c,z_u)
            train_loss.backward()

            opt.step()
            epoch_l += train_loss.item()
           
        epoch_losses.append(epoch_l/len(data))  
        
        if epoch % 10 == 0:
            logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch,epoch_l/len(data) ))

    return epoch_losses

 
def predict_batch(model,x_c1,y,x_c2):
    return model(x_c1,x_c2),y
