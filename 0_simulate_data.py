
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


##### from different single cell and spatial data 
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


####################################################
##### from one single cell without spatial data 
####################################################

wdir = 'node/pancreas/'
sc_ref_path = wdir+'/data/pancreas_sc.h5ad'

rna = an.read_h5ad(sc_ref_path)

sc_exp_shape = rna.X.shape
adata_sc = an.AnnData(X=rna.X)
adata_sc.var_names = rna.var.index.values
adata_sc.obs_names = rna.obs.index.values


adata_sp = an.AnnData(X=rna.X)
adata_sp.var_names = rna.var.index.values
adata_sp.obs_names = ['sp_'+str(x) for x in range(adata_sp.shape[0])]

##now add position and celltype
dfl = pd.read_csv(wdir+'data/pancreas_meta.tsv',sep='\t')
dfl = dfl[['Cell','Celltype (major-lineage)']]
dfl.columns = ['cell','celltype']
adata_sc.obs['celltype'] = pd.merge(rna.obs,dfl, right_on='cell',left_index=True)['celltype'].values


ref_sp = an.read_h5ad('/Users/sishirsubedi/Documents/local_projects/sailr-main/node/sim/data/sim_sp.h5ad')

dfspl = ref_sp.uns['sp_pos'].sample(rna.X.shape[0]).reset_index(drop=True)
dfspl = dfspl[['x','y']]

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(dfspl['x'],dfspl['y'])]

# this celltype assignment is not needed
from sklearn.cluster import KMeans
celltypes = adata_sc.obs['celltype'].unique()
num_celltypes = adata_sc.obs['celltype'].nunique()
kmeans = KMeans(n_clusters=num_celltypes)
kmeans.fit(dfspl)
dfspl['celltype'] = pd.Categorical(kmeans.labels_)
ct_map = {x:y for x,y in zip(dfspl['celltype'].unique(),celltypes)}
dfspl['celltype'] = [ct_map[x] for x in dfspl['celltype']]
adata_sp.uns['celltype'] = dfspl['celltype']


####single cell spatial map
from scipy.spatial.distance import cdist


distmat =  cdist(adata_sc.X.todense(), adata_sp.X.todense())
sorted_indices = np.argsort(distmat, axis=1)
distdf = pd.DataFrame(sorted_indices)

f = [x for x in range(0,25)]
l = [x for x in range(distdf.shape[1]-25,distdf.shape[1])]
distdf = distdf[f+l]

distdf.to_csv(wdir+'results/sc_sp_dist.csv.gz',index=False,compression='gzip')


adata_sp.write(wdir+'data/pancreas_sp.h5ad',compression='gzip')

