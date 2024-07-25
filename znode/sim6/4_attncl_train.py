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


sample = 'sim6'
wdir = 'znode/sim6/'

directory = wdir+'/data'
pattern = 'sim6_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('sim6_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >10:
		break


file_name = file_names[0].replace('.h5ad','').replace('sim6_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)



params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : batch_map[file_name.replace('.h5ad','').replace('sim6_','')].X.shape[1],
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
        'rare_ct_mode' : False, 
      	'num_clusters' : 5, 
        'rare_group_threshold' : 0.1, 
        'rare_group_weight': 2.0,
		'epochs': 1,
		'titration': 15
		}  

picasa_object.estimate_neighbour(params['neighbour_method'])	

def train():
	
	picasa_object.set_nn_params(params)
	picasa_object.train()
	picasa_object.plot_loss()


def eval():
	device = 'cpu'
	picasa_object.set_nn_params(params)
	picasa_object.nn_params['device'] = device
	eval_batch_size = 100
	eval_total_size_per_batch = 10000
	picasa_object.eval_model(eval_batch_size,eval_total_size_per_batch,device)
	picasa_object.save()



train()
eval()
