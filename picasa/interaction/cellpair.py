import numpy as np
import pandas as pd
from anndata import AnnData
from picasa.dutil.data import load_data

def get_nbr_mtx(sc_adata: AnnData, nbr_list: np.array) -> np.array:
    mtx = []
    for nbr_index in nbr_list:
        ctx = load_data(sc_adata,nbr_index,nbr_index+1)
        mtx.append(ctx)
    return np.squeeze(np.array(mtx))

def get_interaction_data(mtx: np.array) -> np.array:
    cpair = []
    for i in range(len(mtx)):
        cpair.append(mtx[i] * np.array([mtx[i] for j, row in enumerate(mtx) if j != i]))
    cpair = np.array(cpair)
    cpair = cpair.reshape(cpair.shape[0]*cpair.shape[1],cpair.shape[2])
    return cpair

def get_interaction_bulk(Z: np.array, mtx: np.array) -> np.array:
    Z = (np.sign(Z) + 1)/2
    df = pd.DataFrame(Z,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    pseudobulk_map = df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']

    pseudobulk = []
    pseudobulk_depth = 1e4
    for _, value in pseudobulk_map.items():        
        pb = mtx[:,value].sum(1)
        pb = (pb/pb.sum()) * pseudobulk_depth
        pseudobulk.append(pb)
    return np.array(pseudobulk)
