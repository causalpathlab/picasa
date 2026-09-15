
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as an
import sys
sys.path.append('/Users/sishirsubedi/Documents/projects/picasa/')

import picasa

wdir = "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/"
samples = ["brca", "lung", "ovary", "pancreas"]
BINNED_NUM = 3000

sample_metadata = {
    "brca": {"label": "Breast (Tumour)", "id_type": "Patient ID"},
    "lung": {"label": "Lung (Tumour)", "id_type": "Patient ID"},
    "ovary": {"label": "Ovary (Tumour)", "id_type": "Patient ID"},
    "pancreas": {"label": "Pancreas (Normal)", "id_type": "Batch ID"},
}

all_sample_records = []

for sample in samples:
    print(f"\n================ Processing Sample: {sample} ================")
    ddir = os.path.join(wdir, sample, "data")
    pattern = f"{sample}_*.h5ad"

    file_paths = glob.glob(os.path.join(ddir, pattern))
    file_names = [os.path.basename(file_path) for file_path in file_paths]

    batch_map = {}
    batch_count = 0
    for file_name in file_names:
        print(f" Loading: {file_name}")
        batch_key = file_name.replace(".h5ad", "").replace(f"{sample}_", "")
        batch_map[batch_key] = an.read_h5ad(os.path.join(ddir, file_name))
        batch_count += 1
        if batch_count >= 12:
            break

    picasa_object = picasa.create_picasa_object(
        batch_map, sample, os.path.join(wdir, sample)
    )

    records = []
    id_label_type = sample_metadata[sample]["id_type"]

    for subject_id in picasa_object.data.adata_list.keys():
        print(f" Processing {id_label_type}: {subject_id}")

        X = picasa_object.data.adata_list[subject_id].X.toarray()
        x_min = X.min()
        x_max = X.max()

        denom = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        X_scaled = (X - x_min) / denom * BINNED_NUM

        X_binned = np.floor(X_scaled).astype(int)
        X_binned = np.clip(X_binned, 0, BINNED_NUM)

        for cell_idx in range(X_binned.shape[0]):
            cell_bins = X_binned[cell_idx, :]
            unique_bins, counts = np.unique(cell_bins, return_counts=True)

            for b, c in zip(unique_bins, counts):
                records.append(
                    {
                        "cell_id": cell_idx,
                        "bin_id": b,
                        "gene_count": c,
                        "subject_id": subject_id,
                        "cancer_sample": sample,
                    }
                )

    df_counts_sample = pd.DataFrame(records)
    all_sample_records.append(df_counts_sample)

# Combine datasets and exclude Bin 0
df_all_counts = pd.concat(all_sample_records, ignore_index=True)
df_expressed = df_all_counts[df_all_counts["bin_id"] > 0]

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
axes = axes.flatten()

panel_letters = ["A", "B", "C", "D"]
for idx, sample in enumerate(samples):
    ax = axes[idx]
    sample_df = df_expressed[df_expressed["cancer_sample"] == sample]
    meta = sample_metadata[sample]

    sns.kdeplot(
        data=sample_df,
        x="bin_id",
        weights="gene_count",
        hue="subject_id",
        fill=False,
        common_norm=False,
        linewidth=1.3,
        alpha=0.85,
        palette="tab10",
        ax=ax,
    )

    ax.set_title(
        f"{panel_letters[idx]}. {meta['label']}",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    
    ax.tick_params(labelleft=True)
    
    ax.set_xlabel(
        "Expression Bin Index (1 – 2,999)", fontweight="bold", fontsize=11
    )
    ax.set_ylabel("Gene Density", fontweight="bold", fontsize=11)
    ax.set_xlim(1, BINNED_NUM)

    if ax.get_legend():
        sns.move_legend(
            ax,
            "upper right",
            title=meta["id_type"],
            fontsize=8.5,
            title_fontsize=9.5,
            frameon=True,
        )

# fig.suptitle(
#     "Supplemental Figure: Model Expression Bin Distributions by Subject/Batch across Cohorts",
#     fontsize=14,
#     fontweight="bold",
#     y=0.98,
# )

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "picasa_bin_dist.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "picasa_bin_dist.png",
    dpi=300,
    bbox_inches="tight",
)
