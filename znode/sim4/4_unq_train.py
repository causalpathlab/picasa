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


sample = 'sim4'
wdir = 'znode/sim4/'

directory = wdir+'/data'
pattern = 'sim4_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('sim4_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >10:
		break


file_name = file_names[0].replace('.h5ad','').replace('sim4_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,'unq',
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('sim4_','')].X.shape[1],
		'embedding_dim' : 1000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,0.0,1.0],
		'temperature_cl' : 1.0,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 0.01,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
      	'loss_clusters' : 5, 
        'loss_threshold' : 0.1, 
        'loss_weight': 2.0,
		'epochs': 1,
		'titration': 15
		}  

picasa_object.estimate_neighbour(params['pair_search_method'])	
picasa_object.set_nn_params(params)
	
unq_layers = [15,15,15]
picasa_object.train_unique(unq_layers,l_rate=0.001,epochs=200,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='unq')

eval_batch_size = 10
eval_total_size = 10000

df_c, df_u,df_batch_id = picasa_object.eval_unique(unq_layers,eval_batch_size, eval_total_size,device='cuda')
df_c.to_csv(wdir+'results/df_c.csv.gz',compression='gzip')
df_u.to_csv(wdir+'results/df_u.csv.gz',compression='gzip')
df_batch_id.to_csv(wdir+'results/df_batch_id.csv.gz',compression='gzip')

