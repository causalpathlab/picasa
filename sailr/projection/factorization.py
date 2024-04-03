import asapc
from sklearn.preprocessing import StandardScaler 		

from ..util.typehint import Adata
from ..dutil import load_data
from typing import Mapping

import logging
logger = logging.getLogger(__name__)


def get_beta(mtx,n_topics):
	nmf_model = asapc.ASAPdcNMF(mtx,n_topics,42)
	nmfres = nmf_model.nmf()

	scaler = StandardScaler()
	beta_log_scaled = scaler.fit_transform(nmfres.beta_log)
	return beta_log_scaled

def predict_theta(mtx,beta):
	pred_model = asapc.ASAPaltNMFPredict(mtx,beta)
	pred = pred_model.predict()
	return pred.corr , pred.theta

def pmf(adata_list: Mapping[str, Adata],
	   ndim: int = 10
	   )-> None:

	adata_rna = adata_list['rna']
	adata_sp = adata_list['spatial']
 
	mtx_sp = load_data(adata_sp,0,adata_sp.shape[0])
	beta = get_beta(mtx_sp.T,ndim)
				 
	corr,theta = predict_theta(mtx_sp.T,beta)
	adata_sp.obsm['X_pmf_corr'] = corr 
	adata_sp.obsm['X_pmf_theta'] = theta 
	
	mtx_rna = load_data(adata_rna,0,adata_rna.shape[0])
	corr, theta = predict_theta(mtx_rna.T,beta)
	adata_rna.obsm['X_pmf_corr'] = corr
	adata_rna.obsm['X_pmf_theta'] = theta
