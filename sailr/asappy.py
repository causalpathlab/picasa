import h5py as hf

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import sailr.preprocessing



############## convert to asap data format

rna = an.read_h5ad('data/sim/sim_sc.h5ad')


fname='data/sim/asapp/data/sim_sc'
row_names = rna.obs.index.values
col_names = rna.var.index.values
smat = rna.X
sailr.preprocessing.read_write.write_h5(fname,row_names,col_names,smat)


############## 

import asappy
sample = 'sim_sc'

wdir = 'data/sim/asapp/'

data_size = 110000
number_batches = 1


asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)

asappy.generate_pseudobulk(asap_object,tree_depth=10)

n_topics = 10 
asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
asappy.generate_model(asap_object)

##############
import anndata as an
asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
spatial = an.read_h5ad('data/sim/sim_sp.h5ad')

import asapc

beta_log_scaled = asap_adata.varm['beta_log_scaled'] 
pred_model = asapc.ASAPaltNMFPredict(spatial.X.T,beta_log_scaled)
pred = pred_model.predict()

sc_corr = pd.DataFrame(asap_adata.obsm['corr'])
sp_corr = pd.DataFrame(pred.corr)


from asappy.util.analysis import quantile_normalization

sc_norm,sp_norm = quantile_normalization(sc_corr.to_numpy(),sp_corr.to_numpy())

sc_norm = pd.DataFrame(sc_norm)
sp_norm = pd.DataFrame(sp_norm)


#### spatial tree and single cell as query

from scipy.spatial.distance import cdist
distmat =  cdist(sc_norm, sp_norm)
sorted_indices = np.argsort(-distmat, axis=1)
distdf = pd.DataFrame(sorted_indices)

f = [x for x in range(0,10)]
l = [x for x in range(distdf.shape[1]-10,distdf.shape[1])]
distdf = distdf[f+l]
distdf.to_csv(wdir+'sc_sp_dist.csv.gz',index=False,compression='gzip')