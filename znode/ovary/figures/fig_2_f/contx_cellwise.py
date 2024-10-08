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
cdir = 'figures/fig_2_f/'


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


import h5py as hf 
from scipy.stats import zscore
from picasa.util.plots import plot_umap_df

picasa_h5 = hf.File(wdir+cdir+'results/picasa_out.h5','r')
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


df_context = pd.DataFrame(np.mean(p1_context, axis=1),index=p1_ylabel)

dfl = pd.read_csv(wdir+cdir+'results/df_umap.csv.gz') 	
dfl.drop(columns=['umap1','umap2'],inplace=True) 


### check heatmap

from sklearn.preprocessing import StandardScaler
def standardize_row(row):
    scaler = StandardScaler()
    row_reshaped = row.values.reshape(-1, 1)  
    row_standardized = scaler.fit_transform(row_reshaped)[:, 0]  
    return pd.Series(row_standardized, index=row.index)
dfh = df_context.apply(standardize_row, axis=1)


sns.clustermap(dfh, cmap='viridis')
plt.tight_layout()
plt.savefig(wdir + cdir+'results/nn__cntx_heatmap.png')
plt.close()


from sklearn.decomposition import PCA

pca = PCA(n_components=15)
principal_components = pca.fit_transform(df_context)

df_context = pd.DataFrame(principal_components,index=df_context.index.values)

dfl = pd.read_csv(wdir+cdir+'results/df_umap.csv.gz') 	
dfl.drop(columns=['umap1','umap2'],inplace=True) 


umap_2d = picasa.ut.analysis.run_umap (df_context.to_numpy(),use_snn=False,min_dist=0.9)

df_umap= pd.DataFrame()
df_umap['cell'] = df_context.index.values

df_umap[['umap1','umap2']] = umap_2d
df_umap['batch'] = [x.split('-')[1].split('_')[0] for x in df_umap['cell'].values]


df_umap = pd.merge(df_umap,dfl,on='cell',how='left')

plot_umap_df(df_umap,'cell_type',wdir+cdir+'results/nn__cntx_',pt_size=1.0,ftype='png') 

