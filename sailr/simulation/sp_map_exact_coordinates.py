import numpy as np
import anndata as an
import pandas as pd 

from .copula import get_simulation_params_from_ref,get_simulated_cells
from ..neighbour import get_NNmodel

SPCODE = ['A','B','C','D']

def assign_spatial_region(vx: int, vy: int, x: int, y: int, b: int)->str:
	if vx < x+b and vy < y+b : return 'A'
	elif vx < x+b and vy > y+b and vy < y+(b*2) : return 'A-B'
	elif vx < x+b and  vy > y+(b*2) : return 'B'
	
	elif vx > x+b and vx < x+(b*2) and vy < y+b  : return 'A-D'
	elif vx > x+b and vx < x+(b*2) and vy > y+b and vy < y+(b*2)  : return 'A-B-C-D'
	elif vx > x+b and vx < x+(b*2) and vy > y+(b*2)  : return 'B-C'

	elif vx > x+(b*2)  and vy < y+b  : return 'D'
	elif vx > x+(b*2)  and vy > y+b and vy < y+(b*2)   : return 'C-D'
	elif vx > x+(b*2)  and vy > y+(b*2)   : return 'C'

def get_spatial_map(sp_ref_path: str) -> pd.DataFrame:
	
	adata = an.read_h5ad(sp_ref_path)
	x = np.array([ float(x.split('x')[0]) for x in adata.obs.position])
	y = np.array([ float(x.split('x')[1]) for x in adata.obs.position])
	xmin = x.min() + 7.5
	ymin = y.min() + 3
	b = 3
	celltypes = [assign_spatial_region(ex,ey,xmin,ymin,b) for ex,ey in zip(x,y)]

	prop_map = {
		'A' : [1.0,0.0,0.0,0.0],
		'B' : [0.0,1.0,0.0,0.0],
		'C' : [0.0,0.0,1.0,0.0],
		'D' : [0.0,0.0,0.0,1.0],
		'A-B' : [1.0,0.0,0.0,0.0],
		'A-D' : [0.0,0.0,0.0,1.0],
		'B-C' : [0.0,1.0,0.0,0.0],
		'C-D' : [0.0,0.0,1.0,0.0],
		'A-B-C-D' : [1.0,0.0,0.0,0.0]
	}


	prop = []
	for i in range(len(celltypes)):prop.append(prop_map[celltypes[i]])
		

	data = pd.DataFrame({
		'x': x,
		'y': y,
		'proportions': prop,
		'celltype': celltypes,
	})

	df_proportions = pd.DataFrame(data['proportions'].values.tolist(), columns=SPCODE)
	df = pd.concat([data[['x', 'y','celltype']], df_proportions], axis=1)
	df.reset_index(inplace=True)
	return df

def assign_sc_to_spatial( 
    sim_params: dict, 
    dfsp: pd.DataFrame, 
    ct_map: dict, 
    sc_size: int,
    rho: float
    ):
			
	all_scs = []
	all_nbrs = []
	for idx in range(dfsp.shape[0]):
	 
		celltype = dfsp.loc[idx,['celltype']].values[0]

		print('generating single cell data for...'+str(idx)+'....'+str(dfsp.shape[0]))

		if len(celltype) == 1:
			selected = get_simulated_cells(sim_params,ct_map[celltype],sc_size,rho)
			all_scs.append(selected.values)
			all_nbrs.append(selected.index.values)
   
		else:
			n_sc = int(sc_size/len(celltype.split('-')))
			scs = []
			nbrs = []
			for ct in celltype.split('-'):
				selected = get_simulated_cells(sim_params,ct_map[ct],n_sc,rho)	
				scs.append(selected.values)
				nbrs.append(selected.index.values)
			scs = np.array(scs)
			scs = scs.reshape((scs.shape[0]*scs.shape[1],scs.shape[2]))
			all_scs.append(scs)
			all_nbrs.append(np.array(nbrs).flatten())
   
	return np.array(all_scs),np.array(all_nbrs)

   
def generate_simdata(
    sc_ref_path: str, 
	sp_ref_path: str, 
	sc_size: int = 16, 
	sc_depth: int = 10000,
	rho: float = 0.9,
	seed: int = 42
    )-> dict:

	sim_params = {}
	get_simulation_params_from_ref(sc_ref_path,sim_params,sc_depth,seed)
	ct_map = {x:y for x,y in zip(SPCODE,sim_params['cts'])}

	dfsp = get_spatial_map(sp_ref_path)

	all_scs, all_nbrs = assign_sc_to_spatial(sim_params,dfsp,ct_map,sc_size,rho)
	
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
	