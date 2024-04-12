

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch

import logging

wdir = 'node/sim/'


logging.basicConfig(filename=wdir+'results/3_pcl_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

rna = an.read_h5ad(wdir+'data/sim_sc.h5ad')

device = 'cpu'
batch_size = 128
input_dims = 17543
latent_dims = 10
encoder_layers = [200,100,10]
projection_layers = [10,25,10]
corruption_rate = 0.001
l_rate = 0.001
epochs= 100

data = sailr.du.nn_load_data(rna,device,batch_size)
features_high = int(data.dataset.vals.max(axis=0).values)
features_low = int(data.dataset.vals.min(axis=0).values)

sailr_model = sailr.nn_pcl.SAILRNET(input_dims, latent_dims, encoder_layers, projection_layers,features_low,features_high,corruption_rate).to(device)
logging.info(sailr_model)

sailr.nn_pcl.train(sailr_model,data,epochs,l_rate)

torch.save(sailr_model.state_dict(),wdir+'results/nn_pcl.model')

