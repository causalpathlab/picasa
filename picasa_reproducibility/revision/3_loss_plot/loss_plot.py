import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)

samples = ["brca", "lung", "ovary"]
wdir = "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/"

sample_metadata = {
    "brca": "Breast (Tumour)",
    "lung": "Lung (Tumour)",
    "ovary": "Ovary (Tumour)",
    "pancreas": "Pancreas (Normal)",
}

fig, axes = plt.subplots(3, 5, figsize=(18, 12), sharex=False)

for row_idx, sample in enumerate(samples):
    print(f"Loading {sample}...")
    file_path_common = os.path.join(
        wdir, sample, "results", "picasa_common_train_loss.txt.gz"
    )
    file_path_unique = os.path.join(
        wdir, sample, "results", "picasa_unique_train_loss.txt.gz"
    )

    df_common = pd.read_csv(file_path_common)
    df_unique = pd.read_csv(file_path_unique)

    row_axes = axes[row_idx]
    tissue_label = sample_metadata[sample]

    row_axes[0].plot(
        df_common.index,
        df_common["ep_cl"],
        label="Common Loss",
        color="#2ca02c",
        lw=1.8,
    )
    row_axes[0].set_title(f"{tissue_label}: Commonality")
    row_axes[0].set_ylabel("Contrastive Loss")

    row_axes[1].plot(
        df_unique.index,
        df_unique["ep_l"],
        label="$L_{total}$",
        color="#d62728",
        lw=1.8,
    )
    row_axes[1].set_title(f"{tissue_label}: $L_{{total}}$")
    row_axes[1].set_ylabel("Total Loss")

    row_axes[2].plot(
        df_unique.index,
        df_unique["el_recon"],
        label="$L_{generative}$",
        color="#1f77b4",
        lw=1.8,
    )
    row_axes[2].set_title(f"{tissue_label}: $L_{{generative}}$")
    row_axes[2].set_ylabel("Recon Loss")

    row_axes[3].plot(
        df_unique.index,
        df_unique["el_batch"],
        label="$L_{predictive}$",
        color="#ff7f0e",
        lw=1.8,
    )
    row_axes[3].set_title(f"{tissue_label}: $L_{{predictive}}$")
    row_axes[3].set_ylabel("Predictive Loss")

    row_axes[4].plot(
        df_unique.index,
        df_unique["el_z"],
        label="$L_{cosine}$",
        color="#9467bd",
        lw=1.8,
    )
    row_axes[4].set_title(f"{tissue_label}: $L_{{cosine}}$")
    row_axes[4].set_ylabel("Cosine Similarity")

    for ax in row_axes:
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        if row_idx == 3:
            ax.set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("picasa_loss.png", dpi=300, bbox_inches="tight")
plt.savefig("picasa_loss.pdf", dpi=300, bbox_inches="tight")
