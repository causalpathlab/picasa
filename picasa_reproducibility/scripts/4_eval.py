import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as an
import pandas as pd
import scanpy as sc
import numpy as np

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SAMPLE = sys.argv[1] 
# WDIR = sys.argv[2]
SAMPLE = 'brca' 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'model_results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_meta_data():
	picasa_adata = an.read_h5ad(os.path.join(DATA_DIR, 'picasa.h5ad'))
	df_meta = picasa_adata.obs.copy()
	df_meta.index = [x.split('@')[0] for x in df_meta.index.values]
	# df_meta.index = ['@'.join(x.split('@')[:2]) for x in df_meta.index.values]
	return df_meta



### graph connectivity
from scipy.sparse.csgraph import connected_components
from scib_metrics.nearest_neighbors import NeighborsResults


def graph_connectivity(X: NeighborsResults, labels: np.ndarray) -> float:

	clust_res = []

	graph = X.knn_graph_distances

	for label in np.unique(labels):
		mask = labels == label
		graph_sub = graph[mask]
		graph_sub = graph_sub[:, mask]
		_, comps = connected_components(graph_sub, connection="strong")
		tab = pd.value_counts(comps)
		clust_res.append(tab.max() / sum(tab))

	return np.mean(clust_res) , np.std(clust_res)

##### LISI

import harmonypy as hm 
def get_metrics_hm(df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):
	
	lisi_res = []
 
	for group in df_meta[constants.GROUP].unique():
		indices = df_meta[df_meta[constants.GROUP] == group].index.values
		if len(indices) >100: 
			res = hm.compute_lisi(df.loc[indices],df_meta.loc[indices],[batch_key])
			lisi_res.append( np.mean(res))
   
	return np.mean(lisi_res),np.std(lisi_res)

from scib_metrics.utils import silhouette_samples
def get_metrics_sil(df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP,chunk_size=1000,rescale=True):
	
	sil_res = []

	X = df.values
	batch = df_meta[batch_key].values
	labels = df_meta[group_key].values
 
	unique_labels = np.unique(labels)
	for group in unique_labels:
		labels_mask = labels == group
		X_subset = X[labels_mask]
		batch_subset = batch[labels_mask]
		n_batches = len(np.unique(batch_subset))

		if (n_batches == 1) or (n_batches == X_subset.shape[0]):
			continue

		sil_per_group = silhouette_samples(X_subset, batch_subset, chunk_size=chunk_size)

		# take only absolute value
		sil_per_group = np.abs(sil_per_group)

		if rescale:
			# scale s.t. highest number is optimal
			sil_per_group = 1 - sil_per_group

		sil_res.append(np.mean(sil_per_group))
 		
	return np.mean(sil_res),np.std(sil_res)


from scib_metrics.nearest_neighbors import pynndescent
from scib_metrics import clisi_knn, nmi_ari_cluster_labels_leiden, silhouette_label


def get_metrics(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    

	avg_res = []
	
	## get global labels and distances
	batch_labels = df_meta[batch_key].values
	group_labels = df_meta[group_key].values				
	neigh_result = pynndescent(df.values,n_neighbors=30)
  
	#### graph connectivity
	graph_res_mean,graph_res_std = graph_connectivity(neigh_result,group_labels)
	
	#### cluster 
	clust_result = nmi_ari_cluster_labels_leiden(neigh_result,group_labels)
 
	ilisi_res_mean,ilisi_res_std = get_metrics_hm(df,df_meta,batch_key,group_key)

	isil_res_mean,isil_res_std = get_metrics_sil(df,df_meta,batch_key,group_key)
  
	clisi_res = clisi_knn(neigh_result,group_labels)
	csil_res = silhouette_label(df.values,group_labels)
 
	avg_res.append([method, graph_res_mean, graph_res_std,ilisi_res_mean,ilisi_res_std,isil_res_mean,isil_res_std,clisi_res,csil_res,clust_result['nmi'],clust_result['ari']])

	df_res = pd.DataFrame(avg_res,columns=['method','graph_mean','graph_std','ilisi_mean','ilisi_std','isil_mean','isil_std','clisi_score','csil_score','nmi','ari'])
 
	return df_res


############## eval 

def eval():
	df_meta = get_meta_data()

	methods = ['pca','combat','harmony', 'scanorama','liger','scvi','dml']

	df_lisi = pd.DataFrame()

	for method in methods:
		print('eval '+method)
		df = pd.read_csv(os.path.join(RESULTS_DIR,'benchmark_'+method+'.csv.gz'))
		df.index = df.iloc[:,0]
		df = df.iloc[:,1:]
		df = df.loc[df_meta.index.values,:]

		df_lisi_res = get_metrics(method,df,df_meta)
		df_lisi = pd.concat([df_lisi,df_lisi_res],axis=0)
		


	### add picasa
	picasa_adata = an.read_h5ad(os.path.join(DATA_DIR, 'picasa.h5ad'))
	df_p = picasa_adata.obsm['common'].copy()
	df_p.index = [x.split('@')[0] for x in df_p.index.values]
	# df_p.index = ['@'.join(x.split('@')[:2]) for x in df_p.index.values]

	df_picasa_lisi = get_metrics('picasa',df_p,df_meta)
	df_lisi = pd.concat([df_lisi,df_picasa_lisi],axis=0)

	df_lisi.to_csv(os.path.join(RESULTS_DIR,'benchmark_all_scores.csv'))



eval()
