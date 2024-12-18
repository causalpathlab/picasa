from . import dutil 
from . import model 
from .util import generate_neighbours
import torch
import logging
import gc
import pandas as pd
import anndata as an
import numpy as np
import itertools



class picasa(object):
    """
    Initialize PICASA model

    Parameters
    ----------
    data: dict of batch_name:anndata, each anndata is separate batch
    pair_mode: 'seq' for generating pairs in sequential - (1,2),(2,3),(3,4)
                else use all possible pair combination
    wdir: path to save model outputs            
    sample: name of data sample            
                
    Returns
    -------
    None

    """
    def __init__(self, 
        data: dutil.data.Dataset, 
        sample: str,
        pair_mode: str,
        wdir: str
        ):
     
     
        self.data = data
        self.sample = sample
        self.wdir = wdir+sample
        
        # dutil.data.create_model_directories(self.wdir,['results'])
        
        logging.basicConfig(filename=self.wdir+'/results/picasa_model.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
  
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')  
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)
        
        
        self.adata_keys = list(self.data.adata_list.keys())

        
        logging.info('Batch pair mode is - '+pair_mode)
        
        if pair_mode == 'seq':
            
            indices = range(len(self.data.adata_list))
            adata_pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 1)]
            adata_pairs.append((adata_pairs[len(adata_pairs)-1][1],adata_pairs[0][0]))
            self.adata_pairs = adata_pairs
            
        elif 'approx' in pair_mode:
            
            from sklearn.decomposition import TruncatedSVD
            from sklearn.metrics.pairwise import euclidean_distances
            
            blist = [self.data.adata_list[x].X for x in self.data.adata_list]
            number_of_pairs = int(pair_mode.split('_')[1])
            blist_keys = self.data.adata_list.keys()

            batch_similarity = euclidean_distances([TruncatedSVD(n_components=10).fit_transform(b).mean(axis=0) for b in blist])

            df_sim = pd.DataFrame(batch_similarity)

            top_n_pairs = pd.DataFrame({
                batch: distances.nsmallest(number_of_pairs + 1).iloc[1:].index.tolist() 
                for batch, distances in df_sim.iterrows()
            }).T

            unique_pairs = set()
            for batch, neighbors in top_n_pairs.iterrows():
                for neighbor in neighbors:
                    unique_pairs.add(tuple(sorted([batch, neighbor])))

            self.adata_pairs = list(unique_pairs)

        else:

            indices = list(range(len(self.data.adata_list)))
            self.adata_pairs = list(itertools.combinations(indices, 2))
                    
    def estimate_neighbour(self,
        method:str ='approx_50'
        ):
     
        logging.info('Pair search method - '+ method)
        self.nbr_map = {}
  
        if 'approx' in method :

            number_of_trees = int(method.split('_')[1])
   
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]
                logging.info('Generating neighbour using approximate method - ANNOY...\n'+p1+'_('+str(self.data.adata_list[p1].X.shape)+') >'+p2+'_('+str(self.data.adata_list[p2].X.shape)+')')
                self.nbr_map[p1+'_'+p2] = generate_neighbours(self.data.adata_list[p2],self.data.adata_list[p1],p1+p2,number_of_trees)
                self.nbr_map[p2+'_'+p1] = generate_neighbours(self.data.adata_list[p1],self.data.adata_list[p2],p2+p1,number_of_trees)
    
        elif method == 'exact':
            from scipy.spatial.distance import cdist
            logging.info('Generating neighbour list using exact method - cdist...')
      
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]

                logging.info(str(self.data.adata_list[p1].X.shape))
                logging.info(str(self.data.adata_list[p2].X.shape))
                distmat =  cdist(self.data.adata_list[p1].X.todense(), self.data.adata_list[p2].X.todense())
                sorted_indices_p1 = np.argsort(distmat, axis=1)
                sorted_indices_p2 = np.argsort(distmat.T, axis=1)
                self.nbr_map[p2+'_'+p1] = {x:y[0] for x,y in enumerate(sorted_indices_p2)}
                self.nbr_map[p1+'_'+p2] = {x:y[0] for x,y in enumerate(sorted_indices_p1)}
                
        logging.info('Pair search estimate is complete.')
                
    def set_nn_params(self,
        params: dict
        ):
        self.nn_params = params
    
    def create_model_adata(self,latent):
    
        batches = latent.keys()
        df = pd.DataFrame()

        for b in batches:
            c_df = latent[b]
            c_df.index = [x+'@'+b for x in c_df.index.values]
            df = pd.concat([df,c_df])

        df.columns = ['common_'+str(x) for x in df.columns]
        
        adata = an.AnnData(obs=pd.DataFrame(index=df.index))
        adata.obsm['common'] = df
        
        batch_loc = len(df.index.values[0].split('@'))-1
        adata.obs['batch'] = [x.split('@')[batch_loc] for x in df.index.values]

        adata.uns['adata_keys'] = self.adata_keys
        adata.uns['adata_pairs'] = self.adata_pairs
        adata.uns['nn_params'] = self.nn_params
        
        batch_ids = {label: idx for idx, label in enumerate(adata.obs['batch'].unique())}
        adata.obs['batch_id'] = [batch_ids[x] for x in adata.obs['batch']]

        nbr_map_df = pd.DataFrame([
            {'batch_pair': l1_item, 'key': k, 'neighbor': v[0], 'score': v[1]}
            for l1_item, inner_map in self.nbr_map.items()
            for k, v in inner_map.items()
        ])
        adata.uns['nbr_map'] = nbr_map_df    
        self.result = adata
            
    def set_batch_mapping(self):
        
        batch_mapping = { idx:label for idx, label in zip(self.result.obs.index.values,self.result.obs['batch_id'])}
        self.batch_mapping = batch_mapping
        
    def set_metadata(self):
        
        df_meta = pd.DataFrame()
        for ad_name in self.data.adata_list:
            ad = self.data.adata_list[ad_name]
            df_meta = pd.concat([df_meta,ad.obs])
        
        df_meta.index = [x+'@'+y for x,y in zip(df_meta.index.values,df_meta['batch'])]
        sel_col = [ x for x in df_meta.columns if x not in ['batch','batch_id']]
        
        self.result.obs = pd.merge(self.result.obs,df_meta[sel_col],left_index=True,right_index=True,how='left')
        
            
    def train_common(self):

        logging.info('Starting PICASA common training...')

        logging.info(self.nn_params)
  
        picasa_model = model.PICASACommonNet(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
        logging.info(picasa_model)
  
        loss = []

        for it in range(self.nn_params['meta_epochs']):
        
            logging.info('meta_epochs : '+ str(it)+'/'+str(self.nn_params['meta_epochs']))
  
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]
    
                logging.info('Training...PICASA common model-'+p1+'_'+p2)
        
                data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],self.nn_params['device'],self.nn_params['batch_size'])

                loss_p1_p2 = model.picasa_train_common(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'])
    
                logging.info('Training...PICASA common model-'+p2+'_'+p1)
    
                data = dutil.nn_load_data_pairs(self.data.adata_list[p2],self.data.adata_list[p1],self.nbr_map[p2+'_'+p1],self.nn_params['device'],self.nn_params['batch_size'])

                loss_p2_p1 = model.picasa_train_common(picasa_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'])

                loss_p1_p2 = np.array(loss_p1_p2)
                loss_p2_p1 = np.array(loss_p2_p1)
                stacked_loss_p = np.vstack((loss_p1_p2, loss_p2_p1))
                loss.append(np.mean(stacked_loss_p, axis=0))

        torch.save(picasa_model.state_dict(),self.wdir+'/results/picasa_common.model')
        pd.DataFrame(loss,columns=['ep_l','cl','al','el']).to_csv(self.wdir+'/results/picasa_common_train_loss.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in '+self.wdir+'/results/picasa_common.model')
    
    def eval_common(self,
        eval_batch_size:int,
        device='cpu'
        ):
     
        picasa_model = model.PICASACommonNet(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers'],self.nn_params['corruption_tol'],self.nn_params['pair_importance_weight']).to(self.nn_params['device'])
  
        picasa_model.load_state_dict(torch.load(self.wdir+'/results/picasa_common.model', map_location=torch.device(device)))
  
        picasa_model.eval()
    
        latent = {}
  
        evaled = []
  
        for ad_pair in list(self.nbr_map.keys()):
            
            if len(ad_pair.split('_')) == 2:
                p1 = ad_pair.split('_')[0]
                p2 = ad_pair.split('_')[1]
            else:
                p1 = ad_pair.split('_')[0]+'_'+ad_pair.split('_')[1]
                p2 = ad_pair.split('_')[2]+'_'+ad_pair.split('_')[3]

            if p1 not in evaled:
                logging.info('eval :'+p1+'_'+p2)
                data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],device,eval_batch_size)
        
                df_latent = pd.DataFrame()

                for x_c1,y,x_c2,nbr_weight in data_pred:
                    picasa_out,ylabel = model.predict_batch_common(picasa_model,x_c1,y,x_c2)
                    df_latent = pd.concat([df_latent,pd.DataFrame(picasa_out.h_c1.cpu().detach().numpy(),index=ylabel)],axis=0)

                    del x_c1, y, x_c2, picasa_out, ylabel
                    gc.collect()
          
                latent[p1] = df_latent
                evaled.append(p1)
                
        self.create_model_adata(latent)
        self.set_metadata()
        

    def train_unique(self,
        input_dim:int,
        enc_layers:list,
        common_latent_dim:int,
        unique_latent_dim:int,
        dec_layers:list,
        l_rate:float,
        epochs:int,
        batch_size:int,
        device:str
        ):

        num_batches = len(self.adata_keys)
        picasa_unq_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
    
        logging.info(picasa_unq_model)

        self.set_batch_mapping()
        
        x_c1_batches = []
        y_batches = []
        x_zc_batches = []
        b_ids_batches = []
  
        logging.info("Creating dataloader for training PICASA unique model.")
  
        for batch in self.adata_keys:
            adata_x = self.data.adata_list[batch]
            df_zcommon = self.result.obsm['common'][self.result.obs['batch']==batch]

            data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',batch_size)
    
            for x_c1,y,x_zc in data:		
                x_c1_batches.append(x_c1)
                y_batches.append(np.array(y))
                x_zc_batches.append(x_zc)
                b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
    
        all_x_c1 = torch.cat(x_c1_batches, dim=0)  
        all_y = np.concatenate(y_batches)        
        all_x_zc = torch.cat(x_zc_batches, dim=0)  
        all_b_ids = torch.cat(b_ids_batches,dim=0)  

        train_data =dutil.get_dataloader_mem(all_x_c1,all_y,all_x_zc,all_b_ids,batch_size,device)

        logging.info('Training... PICASE unique model.')
        loss = model.picasa_train_unique(picasa_unq_model,train_data,l_rate,epochs)

        torch.save(picasa_unq_model.state_dict(),self.wdir+'/results/picasa_unique.model')
        pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'/results/picasa_unique_train_loss.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in '+self.wdir+'/results/picasa_unique.model')
  
    def eval_unique(self,	 
        input_dim:int,
        enc_layers:list,
        common_latent_dim:int,
        unique_latent_dim:int,
        dec_layers:list,
        eval_batch_size:int, 
        device:str
        ):

        num_batches = len(self.adata_keys)
        picasa_unq_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)
    
        picasa_unq_model.load_state_dict(torch.load(self.wdir+'/results/picasa_unique.model', map_location=torch.device(device)))

        picasa_unq_model.eval()
  
        x_c1_batches = []
        y_batches = []
        x_zc_batches = []
        b_ids_batches = []
  
        logging.info("Creating dataloader for evaluating PICASA unique model.")
  
        for batch in self.adata_keys:
            adata_x = self.data.adata_list[batch]
            df_zcommon = self.result.obsm['common'][self.result.obs['batch']==batch]

            data = dutil.nn_load_data_with_latent(adata_x,df_zcommon,batch,'cpu',eval_batch_size)
    
            for x_c1,y,x_zc in data:		
                x_c1_batches.append(x_c1)
                y_batches.append(np.array(y))
                x_zc_batches.append(x_zc)
                b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
    
        all_x_c1 = torch.cat(x_c1_batches, dim=0)  
        all_y = np.concatenate(y_batches)        
        all_x_zc = torch.cat(x_zc_batches, dim=0)  
        all_b_ids = torch.cat(b_ids_batches,dim=0)  

        train_data =dutil.get_dataloader_mem(all_x_c1,all_y,all_x_zc,all_b_ids,eval_batch_size,device)
            
        df_u_latent = pd.DataFrame()
  
        for x_c1,y,x_zc,b_id in train_data:
            z,ylabel = model.predict_batch_unique(picasa_unq_model,x_c1,y,x_zc)
            z_u = z[0]
            df_u_latent = pd.concat([df_u_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel)],axis=0)

        df_u_latent = df_u_latent.loc[self.result.obsm['common'].index.values,:]
        df_u_latent.columns = ['unique_'+str(x) for x in df_u_latent.columns]
        self.result.obsm['unique'] = df_u_latent

    def train_base(self,
        input_dim:int,
        enc_layers:list,
        latent_dim:int,
        dec_layers:list,
        l_rate:float,
        epochs:int,
        batch_size:int,
        device:str
        ):

        num_batches = len(self.adata_keys)
        picasa_base_model = model.PICASABaseNet(input_dim,latent_dim,enc_layers,dec_layers,num_batches).to(device)
    
        logging.info(picasa_base_model)
  
        x_c1_batches = []
        y_batches = []
        b_ids_batches = []
  
        logging.info("Creating dataloader for training PICASA base model.")
  
        for batch in self.adata_keys:
            adata_x = self.data.adata_list[batch]

            data = dutil.nn_load_data_base(adata_x,batch,'cpu',batch_size)
    
            for x_c1,y in data:		
                x_c1_batches.append(x_c1)
                y_batches.append(np.array(y))
                b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
    
        all_x_c1 = torch.cat(x_c1_batches, dim=0)  
        all_y = np.concatenate(y_batches)        
        all_b_ids = torch.cat(b_ids_batches,dim=0)  

        train_data =dutil.get_dataloader_mem_base(all_x_c1,all_y,all_b_ids,batch_size,device)

        logging.info('Training... PICASE base model.')
        loss = model.picasa_train_base(picasa_base_model,train_data,l_rate,epochs)

        torch.save(picasa_base_model.state_dict(),self.wdir+'/results/picasa_base.model')
        pd.DataFrame(loss,columns=['ep_l','el_z','el_recon','el_batch']).to_csv(self.wdir+'/results/picasa_base_train_loss.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in '+self.wdir+'/results/picasa_base.model')
  
    def eval_base(self,	 
        input_dim:int,
        enc_layers:list,
        latent_dim:int,
        dec_layers:list,
        eval_batch_size:int, 
        device:str
        ):

        num_batches = len(self.adata_keys)
        picasa_base_model = model.PICASABaseNet(input_dim,latent_dim,enc_layers,dec_layers,num_batches).to(device)
    
        picasa_base_model.load_state_dict(torch.load(self.wdir+'/results/picasa_base.model', map_location=torch.device(device)))

        picasa_base_model.eval()
  
        x_c1_batches = []
        y_batches = []
        b_ids_batches = []
  
        logging.info("Creating dataloader for evaluating PICASA base model.")
  
        for batch in self.adata_keys:
            adata_x = self.data.adata_list[batch]

            data = dutil.nn_load_data_base(adata_x,batch,'cpu',eval_batch_size)
    
            for x_c1,y in data:		
                x_c1_batches.append(x_c1)
                y_batches.append(np.array(y))
                b_ids_batches.append(torch.tensor([ self.batch_mapping[y_id] for y_id in y]))
    
        all_x_c1 = torch.cat(x_c1_batches, dim=0)  
        all_y = np.concatenate(y_batches)        
        all_b_ids = torch.cat(b_ids_batches,dim=0)  

        train_data =dutil.get_dataloader_mem_base(all_x_c1,all_y,all_b_ids,eval_batch_size,device)
            
        df_u_latent = pd.DataFrame()
  
        for x_c1,y,b_id in train_data:
            z,ylabel = model.predict_batch_base(picasa_base_model,x_c1,y)
            z_u = z[0]
            df_u_latent = pd.concat([df_u_latent,pd.DataFrame(z_u.cpu().detach().numpy(),index=ylabel)],axis=0)

        df_u_base = df_u_latent.loc[self.result.obsm['common'].index.values,:]
        df_u_base.columns = ['base_'+str(x) for x in df_u_latent.columns]
        self.result.obsm['base'] = df_u_base


    def plot_loss(self,
        tag:str
        ):
        from picasa.util.plots import plot_loss
        if tag=='common':
            plot_loss(self.wdir+'/results/picasa_common_train_loss.txt.gz',self.wdir+'/results/picasa_common_train_loss.png')
        elif tag=='unq':
            plot_loss(self.wdir+'/results/picasa_unique_train_loss.txt.gz',self.wdir+'/results/picasa_unique_train_loss.png')
        elif tag=='base':
            plot_loss(self.wdir+'/results/picasa_base_train_loss.txt.gz',self.wdir+'/results/picasa_base_train_loss.png')
    
    def save_model(self):
        self.result.write(self.wdir+'/results/picasa.h5ad',compression='gzip')
        
def create_picasa_object(
    adata_list:an.AnnData,
    sample:str, 
    pair_mode:str,
    wdir:str
    ):
    return picasa(dutil.data.Dataset(adata_list),sample,pair_mode,wdir)
