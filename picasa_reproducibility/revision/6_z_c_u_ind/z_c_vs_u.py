import os
import dcor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler



def compute_hsic(X, Y, gamma=None):
    ##### Gaussian kernel Hilbert-Schmidt Independence Criterion (HSIC)
    n = X.shape[0]
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.exp(-gamma * ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    L = np.exp(-gamma * ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))

    H = np.eye(n) - np.ones((n, n)) / n

    return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def compute_mean_cosine_similarity(X, Y):
    ####Computes mean absolute cosine similarity between matched cell vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    min_dim = min(X.shape[1], Y.shape[1])
    cos_sims = np.abs(np.sum(X_norm[:, :min_dim] * Y_norm[:, :min_dim], axis=1))
    return np.mean(cos_sims)


samples = ["brca", "lung", "ovary", "pancreas"]
wdir = (
    "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/"
)

sample_metadata = {
    "brca": "Breast (Tumour)",
    "lung": "Lung (Tumour)",
    "ovary": "Ovary (Tumour)",
    "pancreas": "Pancreas (Normal)",
}

np.random.seed(42)
n_subsample = 2000  
results = []

for sample in samples:
    print(f"Processing multi-metric evaluation for {sample}...")
    file_path = os.path.join(wdir, sample, "results", "picasa.h5ad")
    picasa_adata = sc.read_h5ad(file_path)

    z_common = picasa_adata.obsm["common"]
    z_unique = picasa_adata.obsm["unique"]

    if hasattr(z_common, "values"):
        z_common = z_common.values
    if hasattr(z_unique, "values"):
        z_unique = z_unique.values

    n_cells = z_common.shape[0]
    if n_cells > n_subsample:
        idx = np.random.choice(n_cells, size=n_subsample, replace=False)
        zc = z_common[idx]
        zu = z_unique[idx]
    else:
        zc = z_common
        zu = z_unique

    zc_scaled = StandardScaler().fit_transform(zc)
    zu_scaled = StandardScaler().fit_transform(zu)

    shuff_idx = np.random.permutation(zc.shape[0])
    zu_shuffled = zu_scaled[shuff_idx]

    obs_dcor = dcor.distance_correlation(zc_scaled, zu_scaled)
    null_dcor = dcor.distance_correlation(zc_scaled, zu_shuffled)

    cca = CCA(n_components=1)
    zc_c, zu_c = cca.fit_transform(zc_scaled, zu_scaled)
    obs_cca = np.abs(np.corrcoef(zc_c[:, 0], zu_c[:, 0])[0, 1])

    zc_c_null, zu_c_null = cca.fit_transform(zc_scaled, zu_shuffled)
    null_cca = np.abs(np.corrcoef(zc_c_null[:, 0], zu_c_null[:, 0])[0, 1])

    mi_matrix_obs = [
        mutual_info_regression(zu_scaled, zc_scaled[:, i])
        for i in range(min(5, zc.shape[1]))
    ]
    obs_mi = np.mean(mi_matrix_obs)

    mi_matrix_null = [
        mutual_info_regression(zu_shuffled, zc_scaled[:, i])
        for i in range(min(5, zc.shape[1]))
    ]
    null_mi = np.mean(mi_matrix_null)

    obs_cos = compute_mean_cosine_similarity(zc_scaled, zu_scaled)
    null_cos = compute_mean_cosine_similarity(zc_scaled, zu_shuffled)

    obs_hsic = compute_hsic(zc_scaled[:500], zu_scaled[:500])  # Compute on subset for speed
    null_hsic = compute_hsic(zc_scaled[:500], zu_shuffled[:500])

    metrics_map = {
        "dCor": (obs_dcor, null_dcor),
        "CCA_Top": (obs_cca, null_cca),
        "Mean_MI": (obs_mi, null_mi),
        "Cosine_Sim": (obs_cos, null_cos),
        "HSIC": (obs_hsic, null_hsic),
    }

    for metric_name, (obs_val, null_val) in metrics_map.items():
        results.append(
            {
                "sample": sample,
                "tissue": sample_metadata[sample],
                "metric": metric_name,
                "observed": obs_val,
                "permuted_baseline": null_val,
            }
        )

df_all_metrics = pd.DataFrame(results)

df_pivot = df_all_metrics.pivot(
    index=["sample", "tissue"],
    columns="metric",
    values=["observed", "permuted_baseline"],
)
print("\n=== MULTI-METRIC INDEPENDENCE RESULTS ===")
print(df_pivot)

####plot
sns.set_theme(style="whitegrid", font_scale=1.0)
df_plot = df_all_metrics.melt(
    id_vars=["tissue", "metric"],
    value_vars=["observed", "permuted_baseline"],
    var_name="Condition",
    value_name="Value",
)

g = sns.catplot(
    data=df_plot,
    x="tissue",
    y="Value",
    hue="Condition",
    col="metric",
    col_wrap=3,
    kind="bar",
    palette={"observed": "#2b5c8f", "permuted_baseline": "#b0b0b0"},
    height=3.5,
    aspect=1.1,
    sharey=False,
    edgecolor="black",
    linewidth=0.8,
)

g.set_titles(col_template="{col_name}", weight="bold")
g.set_xticklabels(rotation=20, ha="right")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(
    "Multi-Metric Disentanglement Comparison Across Latent Spaces",
    weight="bold",
    fontsize=14,
)

plt.savefig("Supplemental_Fig_Multi_Metrics.pdf", bbox_inches="tight")
plt.savefig("Supplemental_Fig_Multi_Metrics.png", dpi=300, bbox_inches="tight")
