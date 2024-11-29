import torch
import logging
logger = logging.getLogger(__name__)

def predict_batch_common(model,x_c1,y,x_c2 ):
    return model(x_c1,x_c2),y

def predict_attention_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    _,x_c1_attention,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_attention

def predict_context_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor                
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    x_c1_context,_,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_context

def get_latent_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)
    x_c1_context,_,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    x_c1_pool_out = model.pooling(x_c1_context)
    h_c1 = model.encoder(x_c1_pool_out)
    return h_c1

def predict_batch_unique(model,x_c1,y,x_zc):
	return model(x_c1,x_zc),y

def predict_batch_base(model,x_c1,y):
	return model(x_c1),y

    
    