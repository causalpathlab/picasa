import os
import anndata as an
import pandas as pd
import numpy as np
import constants 


def get_meta_data(SAMPLE,DATA_DIR):
	picasa_adata = an.read_h5ad(os.path.join(DATA_DIR, 'picasa.h5ad'))
	df_meta = picasa_adata.obs.copy()
	if SAMPLE == 'lung':
		df_meta.index = ['@'.join(x.split('@')[:2]) for x in df_meta.index.values]
	else:
		df_meta.index = [x.split('@')[0] for x in df_meta.index.values]
	return df_meta



##### LISI

import harmonypy as hm 
def get_metrics_hm_batch(df,df_meta,batch_key=constants.BATCH):
	
	lisi_res = hm.compute_lisi(df,df_meta,[batch_key])
	return np.mean(lisi_res),np.std(lisi_res)
   

def get_metrics_hm_group(df,df_meta,group_key=constants.GROUP):
	
	lisi_res = hm.compute_lisi(df,df_meta,[group_key])
	return np.mean(lisi_res),np.std(lisi_res)

def get_metrics_lisi(df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    

	avg_res = []
	
	ilisi_res_mean,ilisi_res_std = get_metrics_hm_batch(df,df_meta,batch_key)
 
	clisi_res_mean,clisi_res_std = get_metrics_hm_group(df,df_meta,group_key)
 
	avg_res.append([ilisi_res_mean,ilisi_res_std,clisi_res_mean,clisi_res_std])

	df_res = pd.DataFrame(avg_res,columns=['ilisi_mean','ilisi_std','clisi_mean','clisi_std'])
	
	return df_res.round(3)





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

	return np.mean(clust_res), np.std(clust_res)



from scib_metrics.utils import silhouette_samples


from scib_metrics.utils import silhouette_samples

def get_silhouette_metric(
    df,
    df_meta,
    mode="batch",  # or "group"
    batch_key=constants.BATCH,
    group_key=constants.GROUP,
    chunk_size=1000,
    rescale=True
):
    sil_res = []

    X = df.values
    batch = df_meta[batch_key].values
    labels = df_meta[group_key].values

    if mode == "batch":
        groups = np.unique(labels)
        for group in groups:
            mask = labels == group
            X_subset = X[mask]
            batch_subset = batch[mask]
            n_batches = len(np.unique(batch_subset))

            if (n_batches == 1) or (n_batches == X_subset.shape[0]):
                continue

            sil = silhouette_samples(X_subset, batch_subset, chunk_size=chunk_size)
            sil = np.abs(sil)
            if rescale:
                sil = 1 - sil
            sil_res.append(np.mean(sil))

    elif mode == "group":
        batches = np.unique(batch)
        for b in batches:
            mask = batch == b
            X_subset = X[mask]
            label_subset = labels[mask]
            n_labels = len(np.unique(label_subset))

            if (n_labels == 1) or (n_labels == X_subset.shape[0]):
                continue

            sil = silhouette_samples(X_subset, label_subset, chunk_size=chunk_size)
            sil = (1+sil)/2
            sil_res.append(np.mean(sil))

    else:
        raise ValueError("mode must be either 'batch' or 'group'")

    return np.mean(sil_res), np.std(sil_res)


from scib_metrics.nearest_neighbors import pynndescent
from scib_metrics import clisi_knn, nmi_ari_cluster_labels_leiden, silhouette_label


def get_metrics_others(df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    

	avg_res = []
	
	## first get nearest neighbours from latent space
	batch_labels = df_meta[batch_key].values
	group_labels = df_meta[group_key].values
	neigh_result = pynndescent(df.values,n_neighbors=30)

	#### graph connectivity 
	graph_cc_mean,graph_cc_std = graph_connectivity(neigh_result,group_labels)
  
 	#### NMI/ARI 
	clust_result = nmi_ari_cluster_labels_leiden(neigh_result,group_labels)
	nmi_score = clust_result['nmi']
	ari_score = clust_result['ari']

	isil_mean,isil_std = get_silhouette_metric(df,df_meta,'batch')
	csil_mean,csil_std = get_silhouette_metric(df,df_meta,'group')

	avg_res.append([graph_cc_mean,graph_cc_std,nmi_score,ari_score,isil_mean,isil_std,csil_mean,csil_std])

	df_res = pd.DataFrame(avg_res,columns=['graphcc_mean','graphcc_std','nmi_score','ari_score','isil_mean','isil_std','csil_mean','csil_std'])
	
	return df_res.round(3)
