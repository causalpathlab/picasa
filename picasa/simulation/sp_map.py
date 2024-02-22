import numpy as np
import anndata as an
import pandas as pd 

from .copula import get_sim_sc_from_ref 
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
		'A-B' : [0.5,0.5,0.0,0.0],
		'A-D' : [0.5,0.0,0.0,0.5],
		'B-C' : [0.0,0.5,0.5,0.0],
		'C-D' : [0.0,0.0,0.5,0.5],
		'A-B-C-D' : [0.25,0.25,0.25,0.25]
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

def assign_sc_to_spatial(dfsc: pd.DataFrame, dfsp: pd.DataFrame,sp_ref_path: str, celltypes: list, ct_map: dict, nbrsize: int)-> list:
	
	model_list = {}
	for ct in celltypes:
		model_ann = get_NNmodel(dfsc[dfsc.index.str.contains(ct)].values)
		model_ann.build()
		model_list[ct] = model_ann

	adata = an.read_h5ad(sp_ref_path)
		
	all_nbrs = []
	for idx,spot in enumerate(adata.X):
     
		celltype = dfsp.loc[idx,['celltype']].values[0]

		if len(celltype) == 1:
			all_nbrs.append(np.array([ model_list[ct_map[celltype]].query(spot,k=nbrsize) ])[0])
   
		else:
			ns = int(nbrsize/len(celltype.split('-')))
			nbrs = []
			for ct in celltype.split('-'):
				nbrs.append(model_list[ct_map[ct]].query(spot,k=ns))
			all_nbrs.append(np.array(nbrs).flatten())
	return all_nbrs

   
def generate_simdata(sc_ref_path: str, 
                    sp_ref_path: str, 
                    sc_size: int = 100, 
                    sc_depth: int = 10000, 
                    seed: int = 42):

    dfsc = get_sim_sc_from_ref(sc_ref_path,sc_size,sc_depth,seed)
    celltypes = pd.Series([x.split('_')[1] for x in dfsc.index.values]).unique()
    ct_map = {x:y for x,y in zip(SPCODE,celltypes)}

    dfsp = get_spatial_map(sp_ref_path)

    nbrs = assign_sc_to_spatial(dfsc,dfsp,sp_ref_path,celltypes,ct_map,nbrsize=16)
    
    return dfsc,dfsp, nbrs
    