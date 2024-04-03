r"""
Type hint definitions
"""

from typing import Union

import anndata as ad
import h5py
import numpy as np
import scipy.sparse

Array = Union[np.ndarray, scipy.sparse.spmatrix]
Backedarray = Union[h5py.Dataset, ad._core.sparse_dataset.SparseDataset]
Adata = ad.AnnData 
