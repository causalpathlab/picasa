from .dutil import Dataset 
from .util.typehint import Adata

import pandas as pd 
import numpy as np

class picasa(object):
	def __init__(self, data: Dataset):
		self.data = data

	def set_spsc_map(self, spsc_map: dict):
		self.spsc_map = spsc_map
  
def create_picasa_object(adata_list: Adata):
	return picasa(Dataset(adata_list))

