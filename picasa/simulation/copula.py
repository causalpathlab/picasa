import numpy as np
import random
import random 
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
import h5py
import anndata as an

from scipy.stats import  poisson


def fit_model(x):
	
	params = []    
	for idx,gene in enumerate(x):
		m = gene.mean()
		v = gene.var()
		
		# try : 
		# 	gene_with_intercept = add_constant(np.zeros_like(gene) + 1, prepend=False)
		# 	nb_model = sm.GLM(gene, gene_with_intercept, family=sm.families.Negativebomial(alpha=1.0)).fit()
		# 	params.append([0.0, nb_model.scale, np.exp(nb_model.params[0])])
		
		# except :
		# 	## fit poisson
		# 	params.append([0.0,1.0,m])
		params.append([0.0,1.0,m])

	return params

def nb_convert_params(mu, theta,epsilon=1e-8):
	"""
	Convert mean/dispersion parameterization of a negative bomial to the ones scipy supports
	See https://en.wikipedia.org/wiki/Negative_bomial_distribution#Alternative_formulations
	"""
	r = theta
	var = mu + 1 / r * mu ** 2
	p = (var - mu) / (var + epsilon)
	return r, 1 - p

# def nb_cdf(counts, mu, theta):
# 	return nbom.cdf(counts, *nb_convert_params(mu, theta))

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
		## from negative bomial
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

def get_simulation_params_from_ref(sc_ref_path,sim_params,depth,seed):

	np.random.seed(seed)
	
	## ref data
	ann = an.read_h5ad(sc_ref_path)
	df = ann.to_df().T
	ct = ann.obs['celltype']
	df.columns = [ x+'@'+y for x,y in zip(df.columns.values,ct)]
	cts = ct.unique()
	
	genes = df.index.values
 
 	## adjust depth
	x_sum = df.values.sum(0)
	df = (df/x_sum)*depth
	x_mean = np.asarray(df.mean(1)).astype('float64')

	rank = 50  

	sim_params['ct_all'] = df
	sim_params['mean_all'] = x_mean 
	sim_params['cts'] = cts
	sim_params['genes'] = genes
	sim_params['rank'] = rank
	sim_params['depth'] = depth
 
	for ct in cts:

		print('generating single cell params for...'+str(ct))

		x_ct = df[[x for x in df.columns if '@'+ct in x]].values
		

		model_params = fit_model(x_ct)
		x_continous = distribution_transformation(model_params,x_ct)

		# # normalization of raw data sample wise
		qt = QuantileTransformer(random_state=0)
		x_all_q = qt.fit_transform(x_continous)

		
		mu_ct = x_continous.mean(1)
		u,d,_ = np.linalg.svd(x_all_q, full_matrices=False)
		L_ct = np.dot(u[:, :rank],np.diag(d[:rank]))  

		sim_params[ct+'_mean'] = mu_ct
		sim_params[ct+'_var'] = L_ct
		
def get_simulated_cells(sim_params,ct,size,rho):
 
	df = sim_params['ct_all']
	x_mean = sim_params['mean_all'] 
	rank = sim_params['rank']
	depth = sim_params['depth']
	genes = sim_params['genes']
 
	all_indx = []
	dfsc = pd.DataFrame()

	x_ct = df[[x for x in df.columns if '@'+ct in x]].values

	L_ct = sim_params[ct+'_var']
	mu_ct = sim_params[ct+'_mean']

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
	sc_global = np.empty_like(sc_ct)
	for i in range(size):
		sc_global[i,:] = np.random.multinomial(depth,sc_prop,1).T.flatten()
	
	sc_all = (rho * sc_ct) + ( (1-rho) * sc_global) 

	## get index ids
	for i in range(size): all_indx.append(str(i) + '_' + ct)

	for i in range(size):
		ct_genes_order = genes[np.argsort(z_ct[:,i])]
		cdf = pd.DataFrame(sc_all[i,:]).T
		cdf.columns = ct_genes_order
		dfsc = pd.concat([dfsc,cdf],axis=0,ignore_index=True)

	print(dfsc.shape)
	dfsc = dfsc.astype(int)
	
	dfsc = dfsc.loc[:,genes]

	dt = h5py.special_dtype(vlen=str) 
	dfsc.index = np.array(np.array(all_indx).flatten(), dtype=dt)

	return dfsc
