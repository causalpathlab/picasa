

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import sailr.model

import logging

logging.basicConfig(filename='3_experiment_nn.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

rna = an.read_h5ad('data/sim/sim_sc.h5ad')
spatial = an.read_h5ad('data/sim/sim_sp.h5ad')
distdf = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/sailr/data/sim/asapp/results/sc_sp_dist.csv.gz')
distdf = distdf[['4991','0']]

device = 'cuda'
batch_size = 256
input_dims = 20309
latent_dims = 10
encoder_layers = [200,200,100,100,10]
projection_layers = [25,10]
kl_lmbda = 0.1
ll_lmbda = 0.001
cl_lmbda = 1.0
l_rate = 0.001
epochs= 1000

data = sailr.du.nn_load_data(rna,spatial,distdf,device,batch_size)

sailr_model = sailr.nn.SAILRNET(input_dims, latent_dims, encoder_layers,projection_layers).to(device)
logging.info(sailr_model)

l1,l2 = sailr.nn.train(sailr_model,data,kl_lmbda, ll_lmbda, cl_lmbda,l_rate,epochs, batch_size)

batch_size=9980
data_pred = sailr.du.nn_load_data(rna,spatial,distdf,device,batch_size)
m = sailr.nn.predict(sailr_model,data_pred)

mats = [
        m.z_sc.cpu().detach().numpy(),
        m.z_spp.cpu().detach().numpy(),
        m.z_spn.cpu().detach().numpy(),
        m.h_sc.cpu().detach().numpy(),
        m.h_spp.cpu().detach().numpy(),
        m.h_spn.cpu().detach().numpy(),
        m.theta.cpu().detach().numpy(),
        m.beta.cpu().detach().numpy(),
        ]
mats_name = ['z_sc','z_spp','z_spn','h_sc','h_spp','h_spn','theta','beta']

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(25, 15))

for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        ax = axes[i, j]  
        print(idx)
        heatmap = ax.imshow(mats[idx], cmap='viridis', aspect='auto')
        ax.set_title(f'{mats_name[idx]}')
        fig.colorbar(heatmap, ax=ax) 

plt.tight_layout()  
plt.savefig('testnn.png');plt.close()