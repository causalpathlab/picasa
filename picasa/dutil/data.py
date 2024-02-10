"""
Base classes for data definition 
"""

from typing import List, Mapping
from ..util.typehint import Adata


def configure_dataset(
    adata : Adata
) -> None:
    pass

    
class Dataset:
    def __init__(self, adata_list : Mapping[str, Adata]) -> None:
        self.adata_list = adata_list
            