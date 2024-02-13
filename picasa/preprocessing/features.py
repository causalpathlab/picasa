import numpy as np


from ..util.typehint import Adata
from typing import Mapping
       
       
def common_features(adata_list: Mapping[str, Adata]):
    
    common_genes = np.intersect1d(adata_list['rna'].var.index.values,adata_list['spatial'].var.index.values)

    adata_list['spatial'].uns['selected_genes'] = np.where(np.isin(adata_list['spatial'].var.index.values,common_genes))[0]

    adata_list['rna'].uns['selected_genes'] = np.where(np.isin(adata_list['rna'].var.index.values,common_genes))[0]
