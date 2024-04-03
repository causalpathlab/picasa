import numpy as np
import anndata as an
import pandas as pd 
from sklearn.cluster import KMeans

from .copula import get_simulation_params_from_ref,get_simulated_cells
from ..neighbour import get_NNmodel


def assign_sc_to_spatial( 
	sim_params: dict, 
	sp_ref_path: str,
	sc_size_per_spot: int,
	num_celltypes: int,
	rho: float
	):
	
	adata = an.read_h5ad(sp_ref_path)
	dfsp = pd.DataFrame(np.array([[float(x.split('x')[0]),float(x.split('x')[1])] for x in adata.obs.position]))

	kmeans = KMeans(n_clusters=num_celltypes)
	kmeans.fit(dfsp)
	dfsp['celltype'] = pd.Categorical(kmeans.labels_	)
 
	ct_map = {x:y for x,y in zip(dfsp['celltype'].unique(),sim_params['cts'])}
 
	dfsp['celltype'] = [ct_map[x] for x in dfsp['celltype']]
 
	all_scs = []
	all_nbrs = []
	for idx in range(dfsp.shape[0]):
	 
		celltype = dfsp.loc[idx,['celltype']].values[0]

		print('generating single cell data for...'+str(idx)+'....'+str(dfsp.shape[0]))

		selected = get_simulated_cells(sim_params,celltype,sc_size_per_spot,rho)
		all_scs.append(selected.values)
		all_nbrs.append(selected.index.values)
      
	return np.array(all_scs),np.array(all_nbrs)

   
def generate_simdata(
	sc_ref_path: str, 
	sp_ref_path: str, 
	sc_size_per_spot: int = 1, 
	num_celltypes: int = 10, 
	sc_depth: int = 10000,
	rho: float = 0.9,
	seed: int = 42
	)-> dict:

	sim_params = {}
	get_simulation_params_from_ref(sc_ref_path,sim_params,sc_depth,seed)

	all_scs, all_nbrs = assign_sc_to_spatial(sim_params,sp_ref_path,sc_size_per_spot,num_celltypes,rho)
	
	### add drop out for spatial
	all_sp = all_scs.sum(axis=1)
	
	ct = []
	for c in dfsp['celltype'].values:
		if len(c) == 1: ct.append(ct_map[c])
		else:
			mix = ''
			for ic in c.split('-'): mix += ct_map[ic] + '-'
			ct.append(mix)    
	dfsp['celltype'] = ct 

	dfsp.columns = [ ct_map[x] if x in ct_map.keys() else x for x in dfsp.columns]
	
	return {'sp_pos': dfsp,
		 	'sp_nbrs': all_nbrs,
		  	'sp_exp': all_sp,
			'sc_exp': all_scs,
			'genes': sim_params['genes']}
	