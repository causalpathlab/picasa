import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.0

df_all_results = pd.read_csv("result.csv")

metrics_def = [
    ("graphcc_mean", "graphcc_std", "Graph Connectivity"),
    ("nmi_score", None, "NMI Score"),
    ("ari_score", None, "ARI Score"),
    ("isil_mean", "isil_std", "iSIL (Batch)"),
    ("csil_mean", "csil_std", "cSIL (Cell Type)"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=300, sharey=True)

x = np.arange(len(metrics_def))
width = 0.35
colors = ["#1f77b4", "#ff7f0e"]  # PICASA (Blue) vs MrVI (Orange)

subsets = [
    (df_all_results[df_all_results["model_space"].str.contains("Common")], "Common Latent Space\n(Evaluated by Cell Type)"),
    (df_all_results[df_all_results["model_space"].str.contains("Unique")], "Unique Latent Space\n(Evaluated by Batch)")
]

for ax, (df_subset, title) in zip(axes, subsets):
    for i, (_, row) in enumerate(df_subset.iterrows()):
        model_name = row["model_space"].split("(")[0].strip()
        offset = width * (i - 0.5)

        means = [row[m[0]] for m in metrics_def]
        stds = [row[m[1]] if (m[1] and not pd.isna(row[m[1]])) else 0 for m in metrics_def]

        ax.bar(x + offset, means, width, yerr=stds, label=model_name, color=colors[i], capsize=3, error_kw={"elinewidth": 1, "capthick": 1})

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics_def], fontsize=9, rotation=15)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

axes[0].set_ylabel("Metric Score", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark_comparison.png", bbox_inches="tight")
plt.savefig("benchmark_comparison.pdf", bbox_inches="tight")
plt.show()