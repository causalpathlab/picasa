
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix 
import scanpy as sc

import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
import constants

def download_file(url,dest_file):
	import requests
	response = requests.get(url, stream=True)
	if response.status_code == 200:
		with open(dest_file, "wb") as file:
			for chunk in response.iter_content(chunk_size=1024):
				file.write(chunk)
		print(f"File downloaded successfully as '{dest_file}'")
	else:
		print(f"Failed to download file. Status code: {response.status_code}")


def prep_sim1_data():
	## splatter data
	wdir = "simulation/"
	i = 1
 
	df= pd.read_csv(wdir+'counts_data_'+str(i)+'.csv').T
	dfl= pd.read_csv(wdir+'col_data_'+str(i)+'.csv')
	print(df.shape, dfl.shape)


	smat = csr_matrix(df.to_numpy())
	adata = ad.AnnData(X=smat)
	adata.var_names = ['g'+str(x)  for x in df.columns.values]
	adata.obs_names = df.index.values

	adata.obs = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell').set_index('Cell')


	column_map = {
		'Cell'  : constants.SAMPLE,
		'Batch' : constants.BATCH,
		'Group' : constants.GROUP,
		}
	adata.obs.rename(columns=column_map,inplace=True)
 
	return adata

def prep_sim2_data():

	file_path = '/data/sishir/data/batch_correction/sim2_multi/sim2_multi_raw.h5ad'
	adata = ad.read(file_path)
	
	adata.obs = adata.obs.iloc[:,:3]
	column_map = {
		'Cell'  : constants.SAMPLE,
		'Batch' : constants.BATCH,
		'Group' : constants.GROUP,
		}
	adata.obs.rename(columns=column_map,inplace=True)

	return adata


def prep_pbmc_data():
			
	df = pd.read_csv('PBMC_60K_CellMetainfo_table.tsv',sep='\t')    
	df = df[['Cell','Celltype (major-lineage)', 'Sample']]
	df.columns = ['cell','celltype','batch']


	import h5py
	f= h5py.File("PBMC_60K_expression.h5", "r") 

	mtx_indptr = f['matrix']['indptr']
	mtx_indices = f['matrix']['indices']
	mtx_data = f['matrix']['data']
	barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
	features = [x.decode('utf-8') for x in f['matrix']['features']['id']]

	rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(barcodes),len(features)))
	mtx= rows.todense()
	barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
	features = [x.decode('utf-8') for x in f['matrix']['features']['id']]
 
	adata = ad.AnnData(X=mtx, obs=barcodes, var=features)
	adata.obs.set_index(0,inplace=True)
	adata.var.set_index(0,inplace=True)
 
	adata.obs = pd.merge(adata.obs,df,left_index=True,right_on='cell')

	adata.obs['batch'] = ['batch_'+str(x) for x in adata.obs['batch']] 

	import scanpy as sc  
	  
	remove_cols = [ x for x in adata.var.index.values if \
		x.startswith('MT-') \
		or x.startswith('RPL') \
		or x.startswith('RPS') \
		or x.startswith('RP1') \
		or x.startswith('MRP')
	]
	
	keep_cols = [ x for x in adata.var.index.values if x  not in remove_cols]
	adata = adata[:,keep_cols]


	sc.pp.filter_genes(adata, min_cells=3)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
 
	# ## use custom hvgenes
	# hvgenes_index = select_hvgenes(adata.X,gene_var_z=2.03)
	# hvgenes = np.array(adata.var.index[hvgenes_index])
	# hvgenes = np.concatenate([hvgenes,marker])
	# hvgenes = np.unique(hvgenes)
	# adata = adata[:, hvgenes] 
 

def prep_pancreas_data():
	file_path = '/data/sishir/data/batch_correction/pancreas/spancreas_raw.h5ad'
	adata = ad.read(file_path)
	
	adata.var.set_index('_index',inplace=True)
	
	adata.obs['Cell'] = adata.obs.index.values
	
	adata.obs = adata.obs[['Cell','BATCH','celltype']]
	column_map = {
		'Cell'  : constants.SAMPLE,
		'BATCH' : constants.BATCH,
		'celltype' : constants.GROUP,
		}
	adata.obs.rename(columns=column_map,inplace=True)

	return adata

def prep_ovary_data():
    '''
    download data from 
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165897
    
    and convert to adata
    '''
	file_path = '/data/sishir/data/ovary/ovary_main.h5ad'
	adata = ad.read(file_path)
	
	ctmap = {
    "T": "T",
    "EOC": "EOC",
    "Macrophages": "Macrophages",
    "CAF": "CAF",
    "B": "B",
    "Mesothelial": "CAF",
    "NK": "NK",
    "DC": "DC",
    "Mast": "Mast",
    "Plasma": "Plasma",
    "pDC": "DC",
    "ILC": "B",
    "Endothelial": "Endothelial"
	}

	adata.obs['celltype'] = [ctmap[x.split('-')[0].split('_')[0]] for x in adata.obs['cell_subtype']]

	adata.obs['celltype'].value_counts()
	
	column_map = {
		'cell'  : constants.SAMPLE,
		'patient_id' : constants.BATCH,
		}
	adata.obs.rename(columns=column_map,inplace=True)

	return adata


def qc(adata):
	import scanpy as sc  
	  
	remove_cols = [ x for x in adata.var.index.values if \
		x.startswith('MT-') \
		or x.startswith('RPL') \
		or x.startswith('RPS') \
		or x.startswith('RP1') \
		or x.startswith('MRP')
	]
	
	keep_cols = [ x for x in adata.var.index.values if x  not in remove_cols]
	adata = adata[:,keep_cols]


	sc.pp.filter_genes(adata, min_cells=3)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata,n_top_genes=2000)
	adata = adata[:, adata.var['highly_variable']]
 
	return adata



def generate_batch_data(adata,sample_name,attr_list):
	batch_keys = list(adata.obs[constants.BATCH].unique())

	for batch in batch_keys:
		adata_c = adata[adata.obs[constants.BATCH].isin([batch])]
		df = adata_c.to_df()

		smat = csr_matrix(df.to_numpy())
		adata_b = ad.AnnData(X=smat)
		adata_b.var_names = df.columns.values
		adata_b.obs_names = df.index.values
	
		for att in attr_list:
			adata_b.obs[att] = adata_c.obs[att].values

		adata_b.write(sample_name+'_'+str(batch)+'.h5ad',compression='gzip')



##get adata
sample = 'sim3'
adata = prep_sim1_data()


sample = 'ovary'
adata = prep_sim1_data()


adata = qc(adata)
attr_list = [constants.BATCH,constants.GROUP,'treatment_phase','cell_type']
generate_batch_data(adata,sample,attr_list)



