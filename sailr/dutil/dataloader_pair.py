from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch
import pytorch_lightning as pl
from scipy import sparse
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SparseData():
	def __init__(self,sc_indptr,sc_indices,sc_vals,sc_shape,sc_label,sp_indptr,sp_indices,sp_vals,sp_shape,scsp_map,sp_x,sp_y):
		self.sc_indptr = sc_indptr
		self.sc_indices = sc_indices
		self.sc_vals = sc_vals
		self.sc_shape = sc_shape
		self.sc_label = sc_label
		self.sp_indptr = sp_indptr
		self.sp_indices = sp_indices
		self.sp_vals = sp_vals
		self.sp_shape = sp_shape
		self.scsp_map = scsp_map
		self.sp_x = sp_x
		self.sp_y = sp_y

class SparseDataset(Dataset):
	def __init__(self, sparse_data,device):
		self.sc_indptr = sparse_data.sc_indptr
		self.sc_indices = sparse_data.sc_indices
		self.sc_vals = sparse_data.sc_vals
		self.sc_shape = sparse_data.sc_shape
		self.sc_label = sparse_data.sc_label
		self.sp_indptr = sparse_data.sp_indptr
		self.sp_indices = sparse_data.sp_indices
		self.sp_vals = sparse_data.sp_vals
		self.sp_shape = sparse_data.sp_shape
		self.scsp_map = sparse_data.scsp_map
		self.sp_x = sparse_data.sp_x
		self.sp_y = sparse_data.sp_y
		self.device = device

	def __len__(self):
		return self.sc_shape[0]

	def __getitem__(self, idx):

		sc_cell = torch.zeros((self.sc_shape[1],), dtype=torch.int32, device=self.device)
		sc_ind1,sc_ind2 = self.sc_indptr[idx],self.sc_indptr[idx+1]
		sc_cell[self.sc_indices[sc_ind1:sc_ind2].long()] = self.sc_vals[sc_ind1:sc_ind2]

		## use sc spatial map to get positive and negative index
		sp_pos_idx = self.scsp_map[idx][0]
  
		sp_pos_cell = torch.zeros((self.sp_shape[1],), dtype=torch.int32, device=self.device)
		sp_pos_ind1,sp_pos_ind2 = self.sp_indptr[sp_pos_idx],self.sp_indptr[sp_pos_idx+1]
		sp_pos_cell[self.sp_indices[sp_pos_ind1:sp_pos_ind2].long()] = self.sp_vals[sp_pos_ind1:sp_pos_ind2]

		return sc_cell, self.sc_label[idx], sp_pos_cell, self.sp_x[idx], self.sp_y[idx]

def nn_load_data(adata_sc,adata_sp,distdf,device,bath_size,bin=100):


	device = torch.device(device)
 
	sc_indptr = torch.tensor(adata_sc.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
	sc_indices = torch.tensor(adata_sc.X.indices.astype(np.int32), dtype=torch.int32, device=device)
	sc_vals = torch.tensor(adata_sc.X.data.astype(np.int32), dtype=torch.int32, device=device)
	sc_shape = tuple(adata_sc.X.shape)
	sc_label = adata_sc.obs.index.values

	# if isinstance(adata_sp.X,np.ndarray):
	spmat = sparse.csr_matrix(adata_sp.X)

	sp_indptr = torch.tensor(spmat.indptr.astype(np.int32), dtype=torch.int32, device=device)
	sp_indices = torch.tensor(spmat.indices.astype(np.int32), dtype=torch.int32, device=device)
	sp_vals = torch.tensor(spmat.data.astype(np.int32), dtype=torch.int32, device=device)
	sp_shape = tuple(adata_sp.X.shape)

	sp_x = np.array([ int(float(x.split('x')[0]))for x in adata_sp.uns['position']])
	sp_y = np.array([ int(float(x.split('x')[1])) for x in adata_sp.uns['position']])
	sp_x_mapped = (sp_x - sp_x.min()) * (bin / (sp_x.max() - sp_x.min())) + 1
	sp_y_mapped = (sp_y - sp_y.min()) * (bin / (sp_y.max() - sp_y.min())) + 1
	sp_x_mapped = torch.tensor(sp_x_mapped.astype(np.int32),dtype=torch.int32, device=device)
	sp_y_mapped = torch.tensor(sp_y_mapped.astype(np.int32),dtype=torch.int32, device=device)

	scsp_map ={x:list(y) for x,y in zip(range(distdf.shape[0]),distdf.values)}

	spdata = SparseData(sc_indptr,sc_indices,sc_vals,sc_shape,sc_label,sp_indptr,sp_indices,sp_vals,sp_shape,scsp_map,sp_x_mapped,sp_y_mapped)

	return DataLoader(SparseDataset(spdata,device), batch_size=bath_size, shuffle=True)

class nn_load_data_mgpu(pl.LightningDataModule):
	
	def __init__(self,adata_sc,adata_sp,distdf,device,batch_size):
		super().__init__()
		self.adata_sc = adata_sc
		self.adata_sp = adata_sp
		self.distdf = distdf
		self.device = device
		self.batch_size = batch_size

	def train_dataloader(self):
		device = torch.device(self.device)
	
		sc_indptr = torch.tensor(self.adata_sc.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
		sc_indices = torch.tensor(self.adata_sc.X.indices.astype(np.int32), dtype=torch.int32, device=device)
		sc_vals = torch.tensor(self.adata_sc.X.data.astype(np.int32), dtype=torch.int32, device=device)
		sc_shape = tuple(self.adata_sc.X.shape)
		sc_label = self.adata_sc.obs.index.values


		sp_indptr = torch.tensor(self.adata_sp.X.indptr.astype(np.int32), dtype=torch.int32, device=device)
		sp_indices = torch.tensor(self.adata_sp.X.indices.astype(np.int32), dtype=torch.int32, device=device)
		sp_vals = torch.tensor(self.adata_sp.X.data.astype(np.int32), dtype=torch.int32, device=device)
		sp_shape = tuple(self.adata_sp.X.shape)

		sp_x = torch.tensor(np.array([ int(x.split('x')[0]) for x in self.adata_sp.uns['position']]),dtype=torch.int32, device=device)
		sp_y = torch.tensor(np.array([ int(x.split('x')[1]) for x in self.adata_sp.uns['position']]),dtype=torch.int32, device=device)

		scsp_map ={x:list(y) for x,y in zip(range(self.distdf.shape[0]),self.distdf.values)}
	
		spdata = SparseData(sc_indptr,sc_indices,sc_vals,sc_shape,sc_label,sp_indptr,sp_indices,sp_vals,sp_shape,scsp_map,sp_x,sp_y)

		return DataLoader(SparseDataset(spdata,device), batch_size=self.batch_size, shuffle=True)

