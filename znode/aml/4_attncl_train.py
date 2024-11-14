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


sample = 'aml'
wdir = 'znode/aml/'

directory = wdir+'/data'
pattern = 'aml_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('aml_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break


file_name = file_names[0].replace('.h5ad','').replace('aml_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	'unq',
	wdir)

params = {
'device': 'cuda', 
'batch_size': 100, 
'input_dim': 2000, 
'embedding_dim': 3000, 
'attention_dim': 15, 
'latent_dim': 15, 
'encoder_layers': [100, 15], 
'projection_layers': [15, 15], 
'learning_rate': 0.001, 
'lambda_loss': [1.0, 0.1, 0.0, 1.0], 
'temperature_cl': 1.0, 
'pair_search_method': 
'approx_50', 
'pair_importance_weight': 0.01, 
'corruption_tol': 5, 
'cl_loss_mode': 'weighted', 
'loss_clusters': 2, 
'loss_threshold': 0.1, 
'loss_weight': 0.5, 
'epochs': 1, 
'titration': 12
}


picasa_object.estimate_neighbour(params['pair_search_method'])	

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
