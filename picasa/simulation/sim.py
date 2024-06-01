import numpy as np
import random
import random 
import pandas as pd

import scipy.sparse
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from ..util._lina import rsvd
from ..dutil.data import write_h5

import glob, os
import gc
import h5py

from scipy.stats import nbinom, norm, poisson
import statsmodels.api as sm
from statsmodels.tools import add_constant



def get_bulkdata(bulk_path):
		
	files = []
	for file in glob.glob(bulk_path):
		files.append(file)
	

	dfall = pd.DataFrame()
	cts = []
	for i,f in enumerate(files):
		print('processing...'+str(f))
		df = pd.read_csv(f)
		df = df[df['Additional_annotations'].str.contains('protein_coding')].reset_index(drop=True)
		df = df.drop(columns=['Additional_annotations'])
		
		ct = os.path.basename(f).split('.')[0].replace('_TPM','')
		cols = [str(x)+'_'+ct for x in range(df.shape[1]-2)]
		df.columns = ['gene','length'] + cols
		
		if i == 0:
			dfall = df
		else:
			dfall = pd.merge(dfall,df,on=['gene','length'],how='outer')
		cts.append(ct)
	return dfall,cts

def fit_model(x):
	
	params = []    
	for idx,gene in enumerate(x):
		m = gene.mean()
		v = gene.var()
		
		# try : 
		# 	gene_with_intercept = add_constant(np.zeros_like(gene) + 1, prepend=False)
		# 	nb_model = sm.GLM(gene, gene_with_intercept, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
		# 	params.append([0.0, nb_model.scale, np.exp(nb_model.params[0])])
		
		# except :
		# 	## fit poisson
		# 	params.append([0.0,1.0,m])
		params.append([0.0,1.0,m])

	return params



def nb_convert_params(mu, theta,epsilon=1e-8):
	"""
	Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports
	See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
	"""
	r = theta
	var = mu + 1 / r * mu ** 2
	p = (var - mu) / (var + epsilon)
	return r, 1 - p

def nb_cdf(counts, mu, theta):
	return nbinom.cdf(counts, *nb_convert_params(mu, theta))

def poisson_cdf(counts,mu):
	return poisson.cdf(counts,mu)

def distribution_transformation(params,x,epsilon=1e-8):
	p, n = x.shape
	u = np.zeros((p, n))

	for iter in range(p):
		param = params[iter]
		gene = x[iter]

		'''
  		gene is not an integer, need to consider both gene and gene - 1 to 
		capture the probability mass that may be spread between two consecutive integer values.
  		'''
		## from negative binomial
		# u1 = nb_cdf(gene, mu=param[2], theta=param[1])
		# u2 = nb_cdf(gene-1, mu=param[2], theta=param[1])

		# # from poisson
		u1 = poisson_cdf(gene, mu=param[2])
		u2 = poisson_cdf(gene-1, mu=param[2])
		
  		# perform linear interpolation between the two CDF values u1 and u2 using the random variable v.
		v = np.random.uniform(size=n)
		r = (v * u2) + ((1 - v) * u1)

		## move down from 1 if too close to 1
		idx_adjust = np.where(1 - r < epsilon)
		r[idx_adjust] = r[idx_adjust] - epsilon
		
  		## move up from 0 if too close to 0
		idx_adjust = np.where(r < epsilon)
		r[idx_adjust] = r[idx_adjust] + epsilon

		u[iter, :] = r

	return u


def simdata_from_bulk_copula(bulk_path,sim_data_path,size,rho,depth,seed,batch=False):

	np.random.seed(seed)
	
	## dice bulk data
	# df,cts = get_bulkdata(bulk_path)
	# cts = ['CD8_NAIVE', 'CD4_NAIVE', 'B_CELL_NAIVE', 'TREG_NAIVE', 'MONOCYTES', 'TH2']
	# nz_cutoff = 10
	# df= df[df.iloc[:,2:].sum(1)>nz_cutoff].reset_index(drop=True)
	# genes = df['gene'].values
	# glens = df['length'].values
	# df = df.drop(columns=['gene','length'])

  
 
	import anndata as an

	adata = an.read_h5ad('data/brca_sc.h5ad')
	df = adata.to_df()
	dfl = pd.read_csv('data/brca_scRNA_celllabels_subset.txt',sep='\t') 
	dfl.columns = ['cell','celltype']
 
	sel_ct_map = {
	'T cells CD8':'Lymphoid_TCD4', 
	'T cells CD4':'Lymphoid_TCD8', 
	'Monocytes and Macrophages':'Myeloid',
	'Epithelial cells':'Epithelial', 
	'NK cells':'Lymphoid_NK', 
	'Fibroblasts':'Fibroblast', 
	'Endothelial cells':'Endothelial',
	'B cells':'Lymphoid_B', 
	'PVL':'PVL', 
	'PCs':'PVL', 
	'Dendritic cells':'Dendritic'
	}
	
	dfl['celltype'] = [sel_ct_map[x] for x in dfl['celltype']]
  
	celltypes = pd.merge(pd.DataFrame(adata.obs_names,columns=['cell']),dfl,on='cell',how='left')['celltype'].values 
	df.index = [str(x)+'_'+y for x,y in enumerate(celltypes)]
	df = df.T
 
	# x_sum = df.values.sum(0)
	# df = (df/x_sum)*depth
	x_mean = df.mean(1)
  
	# cts = ['Endothelial', 'Fibroblast', 'Lymphoid_TCD4','Lymphoid_TCD8','Lymphoid_NK','Lymphoid_B','Myeloid', 'Epithelial','PVL','Dendritic']
	cts = ['Endothelial','Fibroblast','Lymphoid_TCD4','Lymphoid_B','Myeloid','Epithelial']
 
	genes = df.index.values
 
	if batch:

		for ct in cts:

			print('generating single cell data for...'+str(ct))

			x_ct = df[[x for x in df.columns if ct in x]].values
			
	
			model_params = fit_model(x_ct)
			x_continous = distribution_transformation(model_params,x_ct)

			# # normalization of raw data sample wise
			qt = QuantileTransformer(random_state=0)
			x_all_q = qt.fit_transform(x_continous)
	
			rank = 50  
			mu_ct = x_continous.mean(1)
			u,d,_ = np.linalg.svd(x_all_q, full_matrices=False)
			L_ct = np.dot(u[:, :rank],np.diag(d[:rank]))  
	
			## sample mvn of given size with 
			z_ct =  np.dot(L_ct, np.random.normal(size=(rank, size))) +   mu_ct[:, np.newaxis]

			## sample original data by column index
			sc_idx = np.array([[random.randint(0, x_ct.shape[1]-1) for _ in range(x_ct.shape[0])] for _ in range(size)])
	
			sc_ct = np.empty_like(z_ct)
	
			for i in range(size):

				## sample single cell from original data
				sc = x_ct[np.arange(x_ct.shape[0])[:,np.newaxis],sc_idx[i][:, np.newaxis]].flatten()

				## rank gene values
				sc_ct[:,i] = np.sort(sc)

			sc_ct = sc_ct[np.arange(z_ct.shape[0])[:, np.newaxis], np.argsort(z_ct)].T

			sc_prop = np.divide(x_mean, np.sum(x_mean))
			sc_random = np.empty_like(sc_ct)
			for i in range(size):
				sc_random[i,:] = np.random.multinomial(depth,sc_prop,1).T.flatten()
			
			sc_all = (rho * sc_ct) + ((1-rho)*sc_random) 
	
			## get index ids
			all_indx = []
			for i in range(size): all_indx.append(str(i) + '_' + ct)

			dfsc = pd.DataFrame(sc_all,columns=genes)   
			print(dfsc.shape)
			dfsc = dfsc.astype(int)
			
			##shuffle columns to break ranked order
			arr = np.arange(dfsc.shape[1])
			np.random.shuffle(arr)
			dfsc = dfsc.iloc[:,arr] 
		
			smat = scipy.sparse.csr_matrix(dfsc.values)
			dt = h5py.special_dtype(vlen=str) 
			all_indx = np.array(np.array(all_indx).flatten(), dtype=dt) 
			write_h5(sim_data_path+'_'+ct,all_indx,genes,smat)

			##clear memory
			del sc_all
			del dfsc
			del smat
			del all_indx
			gc.collect()
   
	else:	
		dfsc = pd.DataFrame()
		all_indx = []

		for ct in cts:

			print('generating single cell data for...'+str(ct))

			x_ct = df[[x for x in df.columns if ct in x]].values
			
	
			model_params = fit_model(x_ct)
			x_continous = distribution_transformation(model_params,x_ct)

			# # normalization of raw data sample wise
			qt = QuantileTransformer(random_state=0)
			x_all_q = qt.fit_transform(x_continous)
	
			rank = 50  
			mu_ct = x_continous.mean(1)
			u,d,_ = np.linalg.svd(x_all_q, full_matrices=False)
			L_ct = np.dot(u[:, :rank],np.diag(d[:rank]))  
	
			## sample mvn of given size with 
			z_ct =  np.dot(L_ct, np.random.normal(size=(rank, size))) +   mu_ct[:, np.newaxis]

			## sample original data by column index
			sc_idx = np.array([[random.randint(0, x_ct.shape[1]-1) for _ in range(x_ct.shape[0])] for _ in range(size)])
	
			sc_ct = np.empty_like(z_ct)
	
			for i in range(size):

				## sample single cell from original data
				sc = x_ct[np.arange(x_ct.shape[0])[:,np.newaxis],sc_idx[i][:, np.newaxis]].flatten()

				## rank gene values
				sc_ct[:,i] = np.sort(sc)

			sc_ct = sc_ct[np.arange(z_ct.shape[0])[:, np.newaxis], np.argsort(z_ct)].T

			sc_prop = np.divide(x_mean, np.sum(x_mean))
			sc_random = np.empty_like(sc_ct)
			for i in range(size):
				sc_random[i,:] = np.random.multinomial(depth,sc_prop,1).T.flatten()
			
			sc_all = (rho * sc_ct) + ((1-rho)*sc_random) 
	
			## get index ids
			for i in range(size): all_indx.append(str(i) + '_' + ct)

			dfsc = pd.concat([dfsc,pd.DataFrame(sc_all,columns=genes)],axis=0,ignore_index=True)
	
		print(dfsc.shape)
		dfsc = dfsc.astype(int)
		
		##shuffle columns to break ranked order
		arr = np.arange(dfsc.shape[1])
		np.random.shuffle(arr)
		dfsc = dfsc.iloc[:,arr] 
	
		smat = scipy.sparse.csr_matrix(dfsc.values)
		dt = h5py.special_dtype(vlen=str) 
		all_indx = np.array(np.array(all_indx).flatten(), dtype=dt) 
		write_h5(sim_data_path,all_indx,genes,smat)

