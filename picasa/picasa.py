from .dutil import Dataset 
from .util.typehint import Adata

import pandas as pd 
import numpy as np

class picasa(object):
	def __init__(self, data: Dataset):
		self.data = data

	def update_common_features(self):
     
		common_genes = np.intersect1d(self.data.adata_list['rna'].var.index.values,self.data.adata_list['spatial'].var.index.values)

		self.data.adata_list['spatial'].uns['selected_genes'] = np.where(np.isin(self.data.adata_list['spatial'].var.index.values,common_genes))[0]
  
		self.data.adata_list['rna'].uns['selected_genes'] = np.where(np.isin(self.data.adata_list['rna'].var.index.values,common_genes))[0]


def create_picasa_object(adata_list: Adata):
	return picasa(Dataset(adata_list))

