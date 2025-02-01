
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
 
	sc.pp.highly_variable_genes(adata,n_top_genes=1660)
 
	hvgs = adata.var[adata.var['highly_variable']].index.values
	
	sg = pd.read_csv('markers.csv')['genes'].values
 
	allhvgs = np.unique(np.concatenate((hvgs,sg)))
 
	adata = adata[:,adata.var.index.isin(allhvgs)]
 
	return adata



def prep_lung_data():
	import h5py as hf
	f = hf.File('NSCLC_GSE148071_expression.h5')

	mtx_indptr = f['matrix']['indptr']
	mtx_indices = f['matrix']['indices']
	mtx_data = f['matrix']['data']
	barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes']]
	features = [x.decode('utf-8') for x in f['matrix']['features']['id']]

	n_genes = len(features)
	n_cells = len(barcodes)

	matrix = csr_matrix((mtx_data, mtx_indices, mtx_indptr), shape=(n_cells, n_genes))

	adata = ad.AnnData(X=matrix)
	adata.obs.index = barcodes
	adata.var.index = features


	dfl = pd.read_csv('NSCLC_GSE148071_CellMetainfo_table.tsv',sep='\t')

	# patients more than 2.5k cells
	sel_patient = dfl.Patient.value_counts().index[:11]
	dfl = dfl[dfl['Patient'].isin(sel_patient)]

	adata = adata[dfl['Cell'].values,:]

	adata.obs[constants.BATCH] = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell')['Patient'].values
	adata.obs[constants.GROUP] = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell')['Celltype (major-lineage)'].values
	adata.obs['sample'] = pd.merge(adata.obs,dfl,left_index=True,right_on='Cell')['Sample'].values

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
 

	marker = [
		"CD2", "CD3D", "CD3E", "CD3G", "KLRC1", "KLRD1", "NKG7", "CD8A", "CD4", 
		"GNLY", "GZMA", "GZMB", "GZMK", "GZMH", "CCR7", "LEF1", "IL7R", "SELL", 
		"LAG3", "TIGIT", "FOXP3", "IL2RA", "IKZF2", "CTLA4", "ITGAE", "ITGA1", 
		"ZNF683", "CD79A", "CD79B", "MS4A1", "HLA-DRs", "CXCR4", "MZB1", "JCHAIN", 
		"IGHG1", "LYZ", "CSF3R", "S100A8", "S100A9", "FCGR3B", "XCR1", "CLEC9A", 
		"FCER1A", "CD1C", "LAMP3", "FDCSP", "CD68", "MRC1", "CD163", "CD14", "FCN1", 
		"TPSAB1", "TPSB2", "GATA2", "IL3RA", "LILRA4", "CLEC4C", "CLDN5", "PECAM1", 
		"VWF", "DLL4", "KCNE3", "ESM1", "ANGPT2", "ACKR1", "GJA5", "PROX1", "PDPN", 
		"RGS5", "CSPG4", "DCN", "COL1A1", "COL1A2", "ACTA2", "MYH11", "CAPS", "SNTN", 
		"CLDN18", "AQP4", "CAV1", "AGER", "SFTPC", "SFTPA1", "ABCA3", "SCGB1A1", 
		"SCGB3A1", "KRT5", "KRT6A", "KRT14", "FOXJ1", "TPPP3", "PIFO"
	]
 
	hvgs = adata.var[adata.var['highly_variable']].index.values
	 
	allhvgs = np.unique(np.concatenate((hvgs,marker)))
 
	adata = adata[:,adata.var.index.isin(allhvgs)]



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

	adata.write('lung_all.h5ad',compression='gzip')
 
	sc.pp.highly_variable_genes(adata,n_top_genes=2000)
 

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
attr_list = [constants.BATCH,constants.GROUP,'sample']
generate_batch_data(adata,sample,attr_list)



