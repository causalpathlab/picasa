import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler

from ..util.typehint import Adata
from typing import Mapping
from ..dutil import load_data

import logging
logger = logging.getLogger(__name__)

def projection_matrix(depth: int, ndims: int, replicates: int) -> np.array:
    rp_list = []
    for iter_o in range(replicates):
        rp = []
        np.random.seed(iter_o)
        for _ in range(depth):
            rp.append(np.random.normal(size = (ndims,1)).flatten())                      
        rp_list.append(np.asarray(rp))
    return np.array(rp_list)

def weight_adjust_projection_matrix(mtx: np.array, rp_mat_arrs: np.array, weight: str) -> np.array:

    if weight == 'std':
        gene_w = np.std(mtx,axis=1)
    elif weight == 'mean':
        gene_w = np.mean(mtx,axis=1)
    
    rp_mat_w_list = []
    for rp_mat in rp_mat_arrs:rp_mat_w_list.append(rp_mat * gene_w)
    
    rp_mat_w = rp_mat_w_list[0]
    for rp_mat in rp_mat_w_list[1:]:rp_mat_w = np.vstack((rp_mat_w,rp_mat))
    
    return np.array(rp_mat_w)

def get_projection(mtx: np.array, rp_arrs: np.array, rp_weight_adjust: bool, ndim: int) -> np.array:

    logging.info('random projection method.')  
    
    if rp_weight_adjust:
        rp_mat = weight_adjust_projection_matrix(mtx,rp_arrs,weight='std')
    else:
        rp_mat = rp_arrs[0]
        for mat in rp_arrs[1:]:rp_mat = np.vstack((rp_mat,mat))
 
    
    Q = np.dot(rp_mat,mtx)
    _, _, Z = randomized_svd(Q, n_components= ndim, random_state=0)
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z.T)
    
    return Z

def rp(adata_list: Mapping[str, Adata],
       depth: int = 10, 
       replicates: int = 10, 
       batch: int = 10000, 
       rp_weight_adjust: bool = True, 
       ndim: int = 10
       )-> None:

    ndims = len(adata_list['spatial'].uns['selected_genes'])
    
    rp_arrs = projection_matrix(depth,ndims,replicates)
        
    for ad in adata_list:    
        adata = adata_list[ad]
        
        if adata.shape[0]<batch:
            mtx = load_data(adata,0,adata.shape[0])
            adata.obsm['X_rp'] = get_projection(mtx.T,rp_arrs, rp_weight_adjust,ndim)

        else:
            blocks = [(start, min(start + (batch), adata.shape[0])) for start in range(0, adata.shape[0], batch)]
            rmtx = np.empty((0, ndim))
            for b in blocks:
                dmtx = load_data(adata,b[0],b[1])
                cmtx = get_projection(dmtx.T,rp_arrs, rp_weight_adjust,ndim)
                rmtx = np.concatenate((rmtx, cmtx), axis=0)                
            adata.obsm['X_rp'] = rmtx
    
