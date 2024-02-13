"""
Base classes for data definition 
"""

from typing import List, Mapping
from ..util.typehint import Adata

import numpy as np
from scipy.sparse import csr_matrix 


def configure_dataset(
    adata : Adata
) -> None:
    pass

    
class Dataset:
    def __init__(self, adata_list : Mapping[str, Adata]) -> None:
        self.adata_list = adata_list
            


def load_data(adata, start_index: int, end_index: int)-> np.array:
    data = adata.X.data
    indices = adata.X.indices
    indptr = adata.X.indptr
    shape = adata.X.shape
    selected_gene_indices = adata.uns['selected_genes']
    
    mtx = []
    for ci in range(start_index,end_index,1):
        mtx.append(np.asarray(
        csr_matrix((data[indptr[ci]:indptr[ci+1]], 
        indices[indptr[ci]:indptr[ci+1]], 
        np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
        shape=(1,shape[1])).todense()).flatten())
    
    mtx = np.asarray(mtx)
    mtx = mtx[:,selected_gene_indices]
    return mtx
