import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf


def CreateH5ADfromMMF(inpath: str, outpath: str, sample: str) -> None:
	
	import scanpy as sc 
	
	adata = sc.read_10x_mtx(inpath)
	
	adata.write(outpath+sample+'.h5ad',compression='gzip')

def CreateH5ADfromMMF_spatial(inpath: str, outpath: str, sample: str, position_coordinate: str) -> None:
    
	import scanpy as sc 
	
	adata = sc.read_10x_mtx(inpath)

	df_pos = pd.read_csv(position_coordinate,header=False)
	df_pos.columns = [ 'barcode', 'in_tissue', 'array_row', 'array_col','pxl_row_in_fullres','pxl_col_in_fullres']
	adata.uns['position'] = df_pos

	adata.write(outpath+sample+'.h5ad',compression='gzip')


def CreateH5fromMMF(inpath: str, outpath: str, sample: str) -> None:
	
	from scipy.io import mmread
	
	f = hf.File(outpath+sample+'.h5','w')

	mm = mmread(inpath+'/matrix.mtx.gz')
	mtx = mm.todense()
	
	features = pd.read_csv(inpath+'/features.tsv.gz',sep='\t',header=None)
	features = [x+'_'+y for x,y in zip(features[0],features[1])]

	barcodes = list(pd.read_csv(inpath+'/barcodes.tsv.gz',sep='\t',header=None).values.flatten())


	smat = csr_matrix(mtx.T)
	
	grp = f.create_group('matrix')

	grp.create_dataset('barcodes',data=barcodes,compression='gzip')

	grp.create_dataset('features',data=features,compression='gzip')

	grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
	grp.create_dataset('indices',data=smat.indices,compression='gzip')
	grp.create_dataset('data',data=smat.data,dtype=np.int32,compression='gzip')

	arr_shape = np.array([len(barcodes),len(features)])

	grp.create_dataset('shape',data=arr_shape)
	
	f.close()

  
