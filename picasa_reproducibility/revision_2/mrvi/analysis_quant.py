import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import harmonypy as hm
from scipy.sparse.csgraph import connected_components
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent
from scib_metrics.utils import silhouette_samples
from scib_metrics import nmi_ari_cluster_labels_leiden

plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.0

adata_picasa = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/results/picasa.h5ad")
idvals = [x.split('@')[0] for x in adata_picasa.obsm['common'].index.values]
df_picasa_common = pd.DataFrame(adata_picasa.obsm['common'].values, index=idvals)
df_picasa_unique = pd.DataFrame(adata_picasa.obsm['unique'].values, index=idvals)

adata_mrvi = ad.read_h5ad("/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/pancreas/data/all_pancreas.h5ad")
df_mrvi_common = pd.read_csv("mrvi_celltype.csv", index_col=0).reindex(adata_mrvi.obs_names)
df_mrvi_unique = pd.read_csv("mrvi_batch.csv", index_col=0).reindex(adata_mrvi.obs_names)

df_mrvi_common = df_mrvi_common.loc[df_picasa_common.index.values]
df_mrvi_unique = df_mrvi_unique.loc[df_picasa_unique.index.values]

df_meta = adata_picasa.obs.copy()
df_meta.index = idvals

### metrics
def graph_connectivity(X: NeighborsResults, labels: np.ndarray) -> float:
    clust_res = []
    graph = X.knn_graph_distances
    for label in np.unique(labels):
        mask = labels == label
        graph_sub = graph[mask][:, mask]
        if graph_sub.shape[0] == 0:
            continue
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.value_counts(comps)
        clust_res.append(tab.max() / sum(tab))
    return np.mean(clust_res), np.std(clust_res)

def get_silhouette_metric(df, df_meta, mode="batch", batch_key='batch', group_key='celltype', chunk_size=1000, rescale=True):
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
            sil = (1 + sil) / 2
            sil_res.append(np.mean(sil))
    else:
        raise ValueError("mode must be either 'batch' or 'group'")

    return np.mean(sil_res), np.std(sil_res)

def get_metrics_evaluation(df, df_meta, eval_key='celltype', batch_key='batch', group_key='celltype'):
    """
    Evaluates latent space based on target evaluation labels:
    - For Common Space: eval_key = 'celltype' (expects biological conservation)
    - For Unique Space: eval_key = 'batch' (expects batch-specific variation isolation)
    """
    avg_res = []
    eval_labels = df_meta[eval_key].values
    neigh_result = pynndescent(df.values, n_neighbors=30)

    graph_cc_mean, graph_cc_std = graph_connectivity(neigh_result, eval_labels)
  
    clust_result = nmi_ari_cluster_labels_leiden(neigh_result, eval_labels)
    nmi_score = clust_result['nmi']
    ari_score = clust_result['ari']

    isil_mean, isil_std = get_silhouette_metric(df, df_meta, 'batch', batch_key, group_key)
    csil_mean, csil_std = get_silhouette_metric(df, df_meta, 'group', batch_key, group_key)

    avg_res.append([graph_cc_mean, graph_cc_std, nmi_score, ari_score, isil_mean, isil_std, csil_mean, csil_std])
    
    df_res = pd.DataFrame(avg_res, columns=['graphcc_mean', 'graphcc_std', 'nmi_score', 'ari_score', 'isil_mean', 'isil_std', 'csil_mean', 'csil_std'])
    return df_res.round(3)

print("Computing metrics for Common spaces (evaluated on celltype)...")
df_res_picasa_common = get_metrics_evaluation(df_picasa_common, df_meta, eval_key='celltype')
df_res_mrvi_common = get_metrics_evaluation(df_mrvi_common, df_meta, eval_key='celltype')

print("Computing metrics for Unique spaces (evaluated on batch)...")
df_res_picasa_unique = get_metrics_evaluation(df_picasa_unique, df_meta, eval_key='batch')
df_res_mrvi_unique = get_metrics_evaluation(df_mrvi_unique, df_meta, eval_key='batch')

df_res_picasa_common["model_space"] = "PICASA (Common)"
df_res_picasa_unique["model_space"] = "PICASA (Unique)"
df_res_mrvi_common["model_space"] = "MrVI (Common)"
df_res_mrvi_unique["model_space"] = "MrVI (Unique)"

df_all_results = pd.concat([df_res_picasa_common, df_res_picasa_unique, df_res_mrvi_common, df_res_mrvi_unique], ignore_index=True)
cols = ["model_space"] + [col for col in df_all_results.columns if col != "model_space"]
df_all_results = df_all_results[cols]
df_all_results.to_csv("result.csv", index=False)

