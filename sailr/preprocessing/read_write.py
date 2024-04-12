import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
from anndata import AnnData
import scanpy as sc 


def CreateH5ADfromMMF(inpath: str, outpath: str, sample: str, index_filter: str = None) -> None:
	
	
	adata = sc.read_10x_mtx(inpath)
	
	if index_filter: 
		df = pd.DataFrame(adata.X.todense())
		df.index = adata.obs.index.values
		df.columns = adata.var.index.values
		df = df[df.index.str.contains(index_filter)]
		smat = csr_matrix(df.values)
		adata2 = AnnData(X=smat)
		adata2.obs_names = df.index
		adata2.var_names = df.columns
		adata2.write(outpath+sample+'.h5ad',compression='gzip')
	else:
		adata.write(outpath+sample+'.h5ad',compression='gzip')

def CreateH5ADfromMMF_spatial(inpath: str, outpath: str, sample: str, position_coordinate: str) -> None:
    	
	adata = sc.read_10x_mtx(inpath)

	df_pos = pd.read_csv(position_coordinate,header=False)
	df_pos.columns = [ 'barcode', 'in_tissue', 'array_row', 'array_col','pxl_row_in_fullres','pxl_col_in_fullres']
	adata.uns['position'] = df_pos

	adata.write(outpath+sample+'.h5ad',compression='gzip')


def CreateH5fromMMF(inpath: str, outpath: str, sample: str) -> None:
	
	from scipy.io import mmread
	
	f = hf.File(outpath+sample+'.h5ad','w')

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

  
def write_h5(fname,row_names,col_names,smat):

	f = hf.File(fname+'.h5','w')

	grp = f.create_group('matrix')

	grp.create_dataset('barcodes', data = row_names ,compression='gzip')

	grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
	grp.create_dataset('indices',data=smat.indices,compression='gzip')
	grp.create_dataset('data',data=smat.data,compression='gzip')

	data_shape = np.array([len(row_names),len(col_names)])
	grp.create_dataset('shape',data=data_shape)
	
	f['matrix'].create_group('features')
	f['matrix']['features'].create_dataset('id',data=col_names,compression='gzip')

	f.close()
 

def write_h5ad_from_h5(infile,outfile):
	f = hf.File(infile, 'r')
	mtx_indptr = f['matrix']['indptr']
	mtx_indices = f['matrix']['indices']
	mtx_data = f['matrix']['data']
	barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
	features = [x.decode('utf-8') for x in f['matrix']['features']['id']]
	shape = (len(barcodes),len(features))

	matrix = csr_matrix((mtx_data, mtx_indices, mtx_indptr), shape=shape)
	
	adata = AnnData(X=matrix)
	adata.obs_names = barcodes
	adata.var_names = features
 
	adata.write_h5ad(outfile)
