import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os


sample = 'ovary'
wdir = 'znode/ovary/'

directory = wdir+'/data'
pattern = 'ovary_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
    print(file_name)
    batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'data/'+file_name)
    batch_count += 1
    if batch_count >=12:
        break


file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
    batch_map,
    wdir)



params = {'device' : 'cuda',
        'batch_size' : 64,
        'input_dim' : batch_map[file_name.replace('.h5ad','').replace('ovary_','')].X.shape[1],
        'embedding_dim' : 1000,
        'attention_dim' : 15,
        'latent_dim' : 15,
        'encoder_layers' : [100,15],
        'projection_layers' : [15,15],
        'learning_rate' : 0.001,
        'lambda_loss' : [1.0,0.1,1.0],
        'temperature_cl' : 1.0,
        'neighbour_method' : 'approx_50',
         'corruption_rate' : 0.0,
        'pair_importance_weight' : 0.01,
        'rare_ct_mode' : False, 
          'num_clusters' : 5, 
        'rare_group_threshold' : 0.1, 
        'rare_group_weight': 2.0,
        'epochs': 1,
        'titration': 10
        }  

# picasa_object.estimate_neighbour(params['neighbour_method'])	

def plot_attention():

    import h5py as hf
    from scipy.stats import zscore
    
 
    picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
    batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

    # since we have only one pair 
    p1,p2 = batch_keys[0],batch_keys[1]

    adata_p1 = picasa_object.data.adata_list[p1]
    adata_p2 = picasa_object.data.adata_list[p2]
    nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

    device = 'cpu'
    picasa_object.set_nn_params(params)
    picasa_object.nn_params['device'] = device
    eval_batch_size = 10
    eval_total_size = 1000
    p1_attention,p1_ylabel = picasa_object.eval_attention(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)
    
     
    dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz') 	 
    dfl = pd.read_csv(wdir+'results/df_umap.csv.gz') 	 

    adata_p1.obs['celltype']= pd.merge(adata_p1.obs,dfl, left_index=True,right_on='cell')['celltype'].values
    adata_p1.obs['celltype'] = ['c_'+str(x) for x in adata_p1.obs['celltype'].values]

    # adata_p1.obs['celltype']= pd.merge(adata_p1.obs,dfl, left_index=True,right_on='cell')['celltype'].values

    unique_celltypes = adata_p1.obs['celltype'].unique()
    num_celltypes = len(unique_celltypes)
    top_genes = []
    top_n = 10
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        df_attn = df_attn.unstack().reset_index()
        df_attn = df_attn.sort_values(0,ascending=False)
        top_genes.append(df_attn['level_0'].unique()[:top_n])
    top_genes = np.unique(np.array(top_genes).flatten())
 
 
    # cols = 3  
    # rows = int(np.ceil(num_celltypes / cols))

    # fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size as needed

    # axes = axes.flatten()
    # plt.figure(figsize=(50,50))
  
    # for idx, ct in enumerate(unique_celltypes):
    # 	ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
    # 	ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
    # 	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
    # 						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        
    # 	df_attn = df_attn.apply(zscore)
    # 	df_attn[df_attn > 5] = 5
    # 	df_attn[df_attn < -5] = -5
    # 	df_attn = df_attn.loc[:,top_genes]
    # 	df_attn = df_attn.loc[top_genes,:]

    # 	sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
    # 	axes[idx].set_title(f"Clustermap for {ct}")

    # for j in range(idx + 1, rows * cols):
    # 	fig.delaxes(axes[j])

    # plt.tight_layout()
    # plt.savefig(wdir + 'results/sc_attention_allct.png')
    # plt.close()

    # marker = ['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB']	 

    # top_genes = np.concatenate((top_genes,np.array(marker)))

    import networkx as nx
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        
   
        # df_attn = df_attn.apply(zscore)
        # df_attn[df_attn > 5] = 5
        # df_attn[df_attn < -5] = -5
        df_attn = df_attn.stack().reset_index().sort_values(0,ascending=False)
                
        df_attn = df_attn[df_attn['level_0'] != df_attn['level_1']]
        
        df_attn = df_attn.iloc[:10,:]
        
        df_attn = pd.pivot(df_attn,columns='level_0',index='level_1')
        
        # score = df_attn.max().sort_values(ascending=False).values[:10].min()
        # df_attn = df_attn[df_attn > score]

        df_attn = df_attn.dropna(how='all', axis=0)
        df_attn = df_attn.dropna(how='all', axis=1)

        print(df_attn.shape)
        
        df_attn.fillna(0.0,inplace=True)
        # sns.heatmap(df_attn, cmap='viridis')

        G = nx.Graph()
        for row in df_attn.index:
            for col in df_attn.columns:
                if pd.notna(df_attn.loc[row, col]) and df_attn.loc[row, col] != 0:
                    G.add_edge(row, col[1], weight=df_attn.loc[row, col].round(2))

        pos = nx.spring_layout(G)  # Layout for the nodes
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.tight_layout()
        plt.savefig(wdir + 'results/sc_attention_top_'+ct+'.png')
        plt.close()
  
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
        df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
        
        df_attn = df_attn.apply(zscore)
        df_attn[df_attn > 5] = 5
        df_attn[df_attn < -5] = -5

        df_attn = df_attn.loc[:,top_genes]
        df_attn = df_attn.loc[top_genes,:]
  
        sns.clustermap(df_attn, cmap='viridis')
        plt.tight_layout()
        plt.savefig(wdir + 'results/sc_attention_'+ct+'.png')
        plt.close()


    # fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

    # axes = axes.flatten()

    # for idx, ct in enumerate(unique_celltypes):
    # 	ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
    # 	ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
    # 	df_attn = pd.DataFrame(np.mean(p1_attention[ct_yindxs], axis=0),
    # 						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
    # 	df_attn = df_attn.apply(zscore)
    # 	df_attn[df_attn > 5] = 5
    # 	df_attn[df_attn < -5] = -5
    # 	df_attn = df_attn.loc[:,marker]
    # 	df_attn = df_attn.loc[marker,:]

    # 	sns.heatmap(df_attn, cmap='viridis', ax=axes[idx])
    # 	axes[idx].set_title(f"Clustermap for {ct}")

    # for j in range(idx + 1, rows * cols):
    # 	fig.delaxes(axes[j])

    # plt.tight_layout()
    # plt.savefig(wdir + 'results/sc_attention_allct_marker.png')
    # plt.close()

def plot_context():
    
    import h5py as hf 
    from scipy.stats import zscore

    picasa_h5 = hf.File(wdir+'results/picasa_out.h5','r')
    batch_keys = [x.decode('utf-8') for x in picasa_h5['batch_keys']]

    # since we have only one pair 
    p1,p2 = batch_keys[0],batch_keys[1]

    adata_p1 = picasa_object.data.adata_list[p1]
    adata_p2 = picasa_object.data.adata_list[p2]
    nbr_map = {x:y for x,y in list(picasa_h5[p1+'_'+p2])}

    device = 'cpu'
    picasa_object.set_nn_params(params)
    picasa_object.nn_params['device'] = device
    eval_batch_size = 10
    eval_total_size = 3000
    p1_context,p1_ylabel = picasa_object.eval_context(adata_p1,adata_p2,nbr_map,eval_batch_size,eval_total_size,device)
    
    # marker = ['EPCAM','MKI67','CD3D','CD68','MS4A1','JCHAIN','PECAM1','PDGFRB']
    # # marker = top_genes
    # df_context = pd.DataFrame(np.mean(p1_context, axis=0),
    # 						index=adata_p1.var.index.values)
    # df_context = df_context.apply(zscore)
    # df_context[df_context > 1] = 1
    # df_context[df_context < -1] = -1
    # # df_context = df_context.loc[marker,:]
    # sns.clustermap(df_context, cmap='viridis')
    # plt.tight_layout()
    # plt.savefig(wdir + 'results/sc_context_allct.png')
    # plt.close()


    # dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz') 	 
    dfl = pd.read_csv(wdir+'results/df_umap.csv.gz') 	 

    adata_p1.obs['celltype']= pd.merge(adata_p1.obs,dfl, left_index=True,right_on='cell')['cluster'].values
    adata_p1.obs['celltype'] = ['c_'+str(x) for x in adata_p1.obs['celltype'].values]


    unique_celltypes = adata_p1.obs['celltype'].unique()
    num_celltypes = len(unique_celltypes)
 
 
 
    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
        df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values)
        df_context = df_context.apply(zscore)
        df_context.to_csv(wdir+'results/sc_context_'+ct+'.csv.gz',compression='gzip')

    cols = 3  
    rows = int(np.ceil(num_celltypes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))  # Adjust figure size as needed

    axes = axes.flatten()

    for idx, ct in enumerate(unique_celltypes):
        ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
        ct_yindxs = np.where(np.isin(p1_ylabel, ct_ylabel))[0]
        df_context = pd.DataFrame(np.mean(p1_context[ct_yindxs], axis=0),
                            index=adata_p1.var.index.values)
        
        df_context = df_context.apply(zscore)
        df_context[df_context > 5] = 5
        df_context[df_context < -5] = -5
        # df_context = df_context.loc[marker,:]

        sns.heatmap(df_context, cmap='viridis', ax=axes[idx])
        axes[idx].set_title(f"Clustermap for {ct}")

    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(wdir + 'results/sc_context_allct_marker.png')
    plt.close()


plot_attention()
# plot_context()

    