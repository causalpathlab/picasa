import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

samples = ["brca", "lung", "ovary"]
wdir = "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/"

sample_metadata = {
    "brca": "Breast (Tumour)",
    "lung": "Lung (Tumour)",
    "ovary": "Ovary (Tumour)",
    "pancreas": "Pancreas (Normal)",
}

data_dict = {}

for sample in samples:
    print(f"Loading {sample}...")
    file_path = os.path.join(wdir, sample, "results", "picasa.h5ad")
    picasa_adata = sc.read_h5ad(file_path)

    if hasattr(picasa_adata.obsm["common"], "values"):
        z_common = picasa_adata.obsm["common"].values
    else:
        z_common = np.array(picasa_adata.obsm["common"])

    g_min = z_common.min()
    g_max = z_common.max()
    g_mean = z_common.mean()
    zero_pct = (z_common == 0).mean() * 100

    print(f"[{sample.upper()}] Min: {g_min:.2f} | Max: {g_max:.2f} | Mean: {g_mean:.2f} | Sparsity (0s): {zero_pct:.1f}%")

    data_dict[sample] = {
        "z_common": z_common,
        "mean": g_mean,
        "zero_pct": zero_pct,
        "label": sample_metadata[sample],
    }

sns.set_theme(style="whitegrid", font_scale=0.95)

fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(18, 8), sharex="row", sharey="row"
)

for col, sample in enumerate(samples):
    d = data_dict[sample]
    z = d["z_common"]

    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    sns.histplot(
        z.flatten(),
        bins=50,
        kde=False,
        color="#1f77b4",
        ax=ax_top,
        edgecolor="black",
        linewidth=0.5,
    )

    ax_top.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"z = 0 ({d['zero_pct']:.1f}% zero)",
    )
    ax_top.axvline(
        d["mean"],
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean ({d['mean']:.2f})",
    )

    ax_top.set_title(d["label"], fontsize=12, fontweight="bold")
    ax_top.set_xlabel("$z_{common}$ Value", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper right")

    if col == 0:
        ax_top.set_ylabel("Frequency", fontsize=11, fontweight="bold")

    ax_bot.boxplot(
        [z[:, i] for i in range(z.shape[1])],
        patch_artist=True,
        boxprops=dict(facecolor="#a6cee3", color="#1f78b4", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(
            marker=".", markersize=1.5, alpha=0.2, markeredgecolor="none"
        ),
        showfliers=True,
    )

    ax_bot.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax_bot.set_xlabel("Latent Dimension (0–24)", fontsize=10)

    ax_bot.set_xticks(range(1, 26, 5))
    ax_bot.set_xticklabels([str(i) for i in range(0, 25, 5)])

    if col == 0:
        ax_bot.set_ylabel(
            "$z_{common}$ Dimension Distribution",
            fontsize=11,
            fontweight="bold",
        )


plt.tight_layout(rect=[0, 0, 1, 0.96])

outfile_png = os.path.join(
    "picasa_zzero.png"
)
outfile_pdf = os.path.join(
    "picasa_zzero.pdf"
)

plt.savefig(outfile_png, dpi=300, bbox_inches="tight")
plt.savefig(outfile_pdf, dpi=300, bbox_inches="tight")

