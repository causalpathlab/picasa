import h5py as hf

import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import picasa



############## convert to asap data format

rna = an.read_h5ad('znode/pbmc/data/pbmc_pbmc1.h5ad')


fname='znode/pbmc/data/pbmc_sc_1b'
row_names = rna.obs.index.values
col_names = rna.var.index.values
smat = rna.X
picasa.preprocessing.read_write.write_h5(fname,row_names,col_names,smat)


############## 

import asappy 
# sample = 'pbmc_sc_2b'
sample = 'pbmc_sc_1b'

wdir = 'znode/pbmc/asap/'

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


## top 10 main paper
asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
	
cluster_resolution= 0.5 ## paper
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
print(asap_adata.obs.cluster.value_counts())
	
## min distance 0.5 paper
asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.5)

asappy.plot_umap(asap_adata,col='cluster',pt_size=3.0,ftype='png')

asap_adata.obs['batch'] = [x.split('_')[2] for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='batch',pt_size=2.0,ftype='png')

dfl = pd.read_csv(wdir+'data/pbmc_label.csv.gz')
dfl.columns = ['cell','celltype','batch']
asap_adata.obs['celltype'] = pd.merge(asap_adata.obs,dfl, right_on='cell',left_index=True)['celltype'].values
asappy.plot_umap(asap_adata,col='celltype',pt_size=3.0,ftype='png')
	
