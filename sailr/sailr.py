from .dutil import Dataset 
from .util.typehint import Adata

import pandas as pd 
import numpy as np

class sailr(object):
	def __init__(self, data: Dataset):
		self.data = data
  
def create_sailr_object(adata_list: Adata):
	return sailr(Dataset(adata_list))

