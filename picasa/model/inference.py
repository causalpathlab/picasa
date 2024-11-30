import torch
import anndata as an
import numpy as np
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


def eval_attention_common(model, data_loader):
    
    model.eval()

    attn_list = []
    ylabel_list = []

    for x_c1,y,x_c2,nbr_weight in data_loader:
        x_c1_attn = predict_attention_common(model,x_c1,x_c2)
        attn_list.append(x_c1_attn.cpu().detach().numpy())
        ylabel_list.append(y)
            
    attn_list = np.concatenate(attn_list, axis=0)
    ylabel_list = np.concatenate(ylabel_list, axis=0)

    return attn_list,ylabel_list

# def get_attention_common_estimate(self,
#     adata_p1:an.AnnData, 
#     adata_p2:an.AnnData,
#     adata_nbr_map:dict,
#     eval_batch_size:int,
#     eval_total_size:int,
#     device:str='cpu'
#     ):
    
#     picasa_model = model.picasa_model.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])

#     picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
#     picasa_model.eval()

#     data_pred = dutil.nn_load_data_pairs(adata_p1, adata_p2, adata_nbr_map,device,eval_batch_size)

#     global_mean = None
#     y_count = 0

#     for x_c1,y,x_c2,nbr_weight in data_pred:
#         x_c1_attn = model.picasa_model.predict_attention(picasa_model,x_c1,x_c2)
#         batch_mean = x_c1_attn.mean(dim=0) 
#         if global_mean is None:
#             global_mean = batch_mean
#         else:
#             global_mean = ( (global_mean * y_count) + (batch_mean * len(y) )) / (y_count + len(y))
            
#         y_count += len(y)
#         if y_count>eval_total_size:
#             break
        
#         del x_c1_attn, y
#         gc.collect()

#     return global_mean


    # attn_file_p1 = self.wdir+'results/'+p1+'_attention.npz'
    # index_file_p1 = self.wdir+'results/'+p1+'_index.csv.gz'
        
    # np.savez_compressed(attn_file_p1, np.concatenate(attn_list, axis=0))
    # pd.DataFrame(ylabel_list).index.to_series().to_csv(index_file_p1, compression='gzip')

# def eval_context_common(self,
#     adata_p1:an.AnnData, 
#     adata_p2:an.AnnData,
#     adata_nbr_map:dict,
#     eval_batch_size:int,
#     eval_total_size:int,
#     device:str='cpu'
#     ):

#     picasa_model = model.picasa_model.PICASANET(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])

#     picasa_model.load_state_dict(torch.load(self.wdir+'results/nn_attncl.model', map_location=torch.device(device)))
#     picasa_model.eval()

#     data_pred = dutil.nn_load_data_pairs(adata_p1, adata_p2, adata_nbr_map,device,eval_batch_size)

#     context_list = []
#     ylabel_list = []
#     y_count = 0

#     for x_c1,y,x_c2,nbr_weight in data_pred:
#         x_context = model.picasa_model.predict_context(picasa_model,x_c1,x_c2)
#         context_list.append(x_context.cpu().detach().numpy())
#         ylabel_list.append(y)

#         y_count += len(y)
#         if y_count>eval_total_size:
#             break

#         del x_context, y
#         gc.collect()


#     context_list = np.concatenate(context_list, axis=0)
#     ylabel_list = np.concatenate(ylabel_list, axis=0)

#     return context_list,ylabel_list  

