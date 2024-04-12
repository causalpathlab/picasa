

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch

import logging

wdir = 'node/sim/'


logging.basicConfig(filename=wdir+'results/3_etm_train.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

rna = an.read_h5ad(wdir+'data/sim_sc.h5ad')

device = 'cpu'
batch_size = 128
input_dims = 20309
latent_dims = 10
encoder_layers = [200,100,10]
l_rate = 0.01
epochs= 500

data = sailr.du.nn_load_data(rna,device,batch_size)

sailr_model = sailr.nn_etm.SAILRNET(input_dims, latent_dims, encoder_layers).to(device)
logging.info(sailr_model)

l1,l2 = sailr.nn_etm.train(sailr_model,data,epochs,l_rate)

torch.save(sailr_model.state_dict(),wdir+'results/nn_etm.model')

