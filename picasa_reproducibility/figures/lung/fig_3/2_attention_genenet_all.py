import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as an
import pandas as pd


sample ='lung'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

############ read original data as adata list
import os 
import glob 

ddir = pp+sample+'/data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

picasa_data = batch_map


############ read model results as adata 
wdir = pp+sample
picasa_adata = an.read_h5ad(wdir+'/results/picasa.h5ad')




from picasa import model,dutil
import torch

nn_params = picasa_adata.uns['nn_params']
picasa_common_model = model.PICASACommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers'],nn_params['corruption_tol'],nn_params['pair_importance_weight']).to(nn_params['device'])
picasa_common_model.load_state_dict(torch.load(wdir+'/results/picasa_common.model', map_location=torch.device(nn_params['device'])))


# p1 = picasa_adata.uns['adata_keys'][0]
# p2 = picasa_adata.uns['adata_keys'][1]


p1 = 'P6'
p2 = 'P3'

adata_p1 = picasa_data[p1]
adata_p2 = picasa_data[p2]
df_nbr = picasa_adata.uns['nbr_map']
df_nbr = df_nbr[df_nbr['batch_pair']==p1+'_'+p2]
nbr_map = {x:(y,z) for x,y,z in zip(df_nbr['key'],df_nbr['neighbor'],df_nbr['score'])}

data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map,'cpu',batch_size=10)
eval_total_size=1000
main_attn,main_y = model.eval_attention_common(picasa_common_model,data_loader,eval_total_size)


##############################################


import numpy as np 

wdir = wdir + '/fig_2/'

unique_celltypes = adata_p1.obs['celltype'].unique()
num_celltypes = len(unique_celltypes)

def get_top_genes(main_attn,main_y,top_n):
    top_genes = []
    ct_ylabel = adata_p1.obs.index.values
    ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]
    df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
                        index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
    np.fill_diagonal(df_attn.values, 0)
    ## note that prev column is now first column 
    df_attn = df_attn.unstack().reset_index()
    df_attn = df_attn.sort_values(0,ascending=False)
    df_attn = df_attn.iloc[:top_n,:]
    top_genes= [x+'_'+y for x,y in zip(df_attn['level_0'],df_attn['level_1'])]
    
    # df_attn['gpair'] = [x+'/'+y for x,y in zip(df_attn['level_0'],df_attn['level_1'])]        
    # top_genes[ct] = df_attn['gpair'].values
        
    return top_genes
        

top_n=2000
top_genes = get_top_genes(main_attn,main_y,top_n)


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap



G = nx.Graph()

for gene in top_genes:
    G.add_node(gene.split('_')[0])
    G.add_node(gene.split('_')[1])
    G.add_edge(gene.split('_')[0], gene.split('_')[1])


node_colors = []
for node in G.nodes():
    node_colors.append('yellowgreen') 

# pos = nx.spring_layout(G, seed=42, k=.1)  
pos = nx.kamada_kawai_layout(G,scale=.1)

plt.figure(figsize=(14, 10))
nx.draw(
    G, pos, with_labels=False, node_size=400, font_size=10,  
    node_color=node_colors, edge_color='lightgray', font_weight='bold'
)

for node, (x, y) in pos.items():
    plt.text(
        x, y + 0.0,  
        s=node, fontsize=8, ha='center', va='center',
        weight='bold'
        # bbox=dict(facecolor='white', edgecolor='none', alpha=0.6)  # Add background for clarity
    )

    
plt.scatter([], [], color='yellowgreen', label='Genes')
plt.legend(title="Cell Types", loc='upper right', fontsize=9)

plt.title("Gene-Gene Network for 10 Cell Types", fontsize=16)

plt.savefig(wdir+'/results/attention_genenet_all.png')
plt.close()

