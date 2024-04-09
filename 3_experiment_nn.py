

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import sailr.model


rna = an.read_h5ad('data/sim/sim_sc.h5ad')
spatial = an.read_h5ad('data/sim/sim_sp.h5ad')
distdf = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/sailr/data/sim/asapp/results/sc_sp_dist.csv.gz')
distdf = distdf[['4991','0']]

device = 'cuda'
batch_size = 1000
input_dims = 20309
latent_dims = 10
layers = [100,50,10]

data = sailr.du.nn_load_data(rna,spatial,distdf,device,batch_size)

sailr_model = sailr.nn.SAILRNET(input_dims, latent_dims, layers).to(device)
print(sailr_model)

m = sailr.nn.train(sailr_model,data,epochs=100,l_rate=0.0001,batch_size=1000)

