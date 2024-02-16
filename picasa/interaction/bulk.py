from anndata import AnnData
import numpy as np
import pandas as pd

import threading
import queue

from ..projection.randprojection import projection_matrix, get_projection

from .cellpair import get_nbr_mtx, get_interaction_data, get_interaction_bulk

def get_pairwise_interaction_batch(
    spatial_index: int, 
    nbr_list: np.array, 
    sc_adata: AnnData,
    rp_arrs: np.array, 
    bulk_result: queue.Queue,
    lock: threading.Lock, 
    sema: threading.Semaphore) -> None:
    
    sema.acquire()
    lock.acquire()
    sc_mtx = get_nbr_mtx(sc_adata,nbr_list)
    lock.release()
    
    cellpair_mtx = get_interaction_data(sc_mtx)
    
    rp_weight_adjust = True
    ndim = 3
    cpair_rp = get_projection(cellpair_mtx.T,rp_arrs, rp_weight_adjust,ndim)

    cpair_bulk = get_interaction_bulk(cpair_rp, sc_mtx)
    
    print(spatial_index,cpair_bulk.shape)
    
    bulk_result.put({spatial_index:cpair_bulk})
    
    sema.release()
        
    
def get_pairwise_bulk_interaction(spsc_map: dict, sc_adata: AnnData):

    depth=10
    replicates=10
    maxthreads=8

    feature_dim = len(sc_adata.uns['selected_genes'])
    rp_arrs = projection_matrix(depth,feature_dim,replicates)

    threads = []
    bulk_result = queue.Queue()
    lock = threading.Lock()
    sema = threading.Semaphore(value=maxthreads)

    for sp_nbr in spsc_map.items():
        thread = threading.Thread(target=get_pairwise_interaction_batch, args=(sp_nbr[0],sp_nbr[1],sc_adata, rp_arrs, bulk_result,lock,sema))
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    bulk_data = []
    while not bulk_result.empty():
        bulk_data.append(bulk_result.get())

    return np.array(bulk_data)

