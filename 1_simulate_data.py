
import anndata as an
import pandas as pd
import numpy as np
import sailr
from scipy.sparse import csr_matrix 

import logging

logging.basicConfig(filename='1_simdata.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')
  
sc_ref_path = 'data/sim/brcasim_sc.h5ad'
sp_ref_path = 'data/sim/brcasim_sp.h5ad'

num_celltype = 7
sc_size_per_spot = 2
simdata = sailr.sim.generate_simdata(sc_ref_path,sp_ref_path,num_celltype,sc_size_per_spot)



##check spatial labels
import matplotlib.pylab as plt
import seaborn as sns
sns.scatterplot(x='x',y='y',hue='celltype',data=simdata['sp_pos'])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('test.png');plt.close()

sc_exp_shape = simdata['sc_exp'].shape
adata_sc = an.AnnData(X=csr_matrix(simdata['sc_exp'].reshape(
(sc_exp_shape[0] * sc_exp_shape[1],sc_exp_shape[2]))))
adata_sc.var_names = simdata['genes']
adata_sc.obs_names = simdata['sp_nbrs'].reshape((sc_exp_shape[0] * sc_exp_shape[1],-1)).flatten()


adata_sp = an.AnnData(X=simdata['sp_exp'])
adata_sp.var_names = simdata['genes']
adata_sp.obs_names = ['sp_'+str(x) for x in range(adata_sp.shape[0])]
adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(simdata['sp_pos']['x'],simdata['sp_pos']['y'])]


cell_indx = []
t = [ x for x in range(sc_size_per_spot)]
for si in range(simdata['sp_nbrs'].shape[0]):
    cell_indx.append(np.array([x + (sc_size_per_spot * si) for x in t]))
    
spsc_map ={'sp_'+str(x):list(y) for x,y in zip(range(simdata['sp_nbrs'].shape[0]),cell_indx)}

adata_sp.uns['spsc_map'] = spsc_map
adata_sp.uns['sp_pos'] = simdata['sp_pos']
adata_sp.uns['genes'] = simdata['genes']

adata_sp.write('data/sim/sim_sp.h5ad',compression='gzip')
adata_sc.write('data/sim/sim_sc.h5ad',compression='gzip')

