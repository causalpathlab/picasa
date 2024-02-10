from .dutil import Dataset 
from . import genomics
from .util.typehint import Adata

import pandas as pd 
import numpy as np

class picasa(object):
	def __init__(self, data: Dataset):
		self.data = data
	
	def annotate_features(self, gtf: str, gtf_by: str):
	
		gtf = genomics.read_gtf(gtf).query("feature == 'gene'").split_attribute()

		gtf = gtf.sort_values("seqname").drop_duplicates(subset=[gtf_by], keep="last")

		merge_df = pd.concat([
			pd.DataFrame(gtf.to_bed(name=gtf_by)),
			pd.DataFrame(gtf).drop(columns=genomics.Gtf.COLUMNS)  # Only use the splitted attributes
		], axis=1).set_index(gtf_by)

		merge_df.index = merge_df.index.str.upper()

		self.data.adata_list['rna'].var = pd.merge(self.data.adata_list['rna'].var,merge_df,left_index=True,right_index=True,how='left')
		self.data.adata_list['spatial'].var = pd.merge(self.data.adata_list['spatial'].var,merge_df,left_index=True,right_index=True,how='left')

	def update_common_features(self):
     
		common_genes = np.intersect1d(self.data.adata_list['rna'].var.index.values,self.data.adata_list['spatial'].var.index.values)

		self.data.adata_list['spatial'].uns['selected_genes'] = np.where(np.isin(self.data.adata_list['spatial'].var.index.values,common_genes))[0]
  
		self.data.adata_list['rna'].uns['selected_genes'] = np.where(np.isin(self.data.adata_list['rna'].var.index.values,common_genes))[0]


def create_picasa_object(adata_list: Adata):
	return picasa(Dataset(adata_list))

