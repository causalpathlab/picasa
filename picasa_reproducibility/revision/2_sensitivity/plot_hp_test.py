import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


WDIR1 = "/Users/sishirsubedi/Documents/projects/picasa/revision/2_sensitivity/"

SAMPLE_LIST_1 = [
    "sim1_rep1_emb10",
    "sim1_rep1_emb50",
    "sim1_rep1_emb100",
    "sim1_rep1_emb250",
    "sim1_rep1_emb500",
    "sim1_rep1_emb1000",
    "sim1_rep1_emb2500",
    "sim1_rep1_emb5000",
    "sim1_rep1_emb10000",
    "sim1_rep1_attn8",
    "sim1_rep1_attn16",
    "sim1_rep1_attn32",
    "sim1_rep1_attn64",
    "sim1_rep1_latn8",
    "sim1_rep1_latn16",
    "sim1_rep1_latn32",
    "sim1_rep1_latn64",
]

df_list = []
for SAMPLE in SAMPLE_LIST_1:
    CWDIR = os.path.join(WDIR1, SAMPLE)
    dfc = pd.read_csv(os.path.join(CWDIR, "results/benchmark_scores.csv.gz"))
    dfc["exp"] = SAMPLE.split("_")[2]
    df_list.append(dfc)

WDIR2 = "/Users/sishirsubedi/Documents/projects/picasa/revision/2_sensitivity_v2/"

SAMPLE_LIST_2 = [
    "pancreas_rep1_emb500",
    "pancreas_rep1_emb1000",
    "pancreas_rep1_emb2000",
    "pancreas_rep1_emb3000",
    "pancreas_rep1_emb4000",
    "pancreas_rep1_emb5000",
]

for SAMPLE in SAMPLE_LIST_2:
    CWDIR = os.path.join(WDIR2, SAMPLE)
    dfc = pd.read_csv(os.path.join(CWDIR, "results/benchmark_scores.csv.gz"))
    # Fixed prefix matching to alignment key "pemb"
    dfc["exp"] = "p" + SAMPLE.split("_")[2]
    df_list.append(dfc)

df = pd.concat(df_list, ignore_index=True)

OUTFILE_PNG = os.path.join(WDIR2, "results", "picasa_sensitivity_analysis.png")
OUTFILE_PDF = os.path.join(WDIR2, "results", "picasa_sensitivity_analysis.pdf")
os.makedirs(os.path.dirname(OUTFILE_PNG), exist_ok=True)


groups = {
    "Vocabulary size (Sim1)": "emb",
    "Attention dimension (Sim1)": "attn",
    "Latent dimension (Sim1)": "latn",
    "Vocabulary size (Normal-Pancreas)": "pemb",
}

data = {}

for group_name, prefix in groups.items():
    d = df[df["exp"].str.startswith(prefix)].copy()
    d["dim"] = d["exp"].str.replace(prefix, "", regex=False).astype(int)
    d = d.sort_values("dim")
    data[group_name] = d

def format_k(x, pos):
    if x >= 1000:
        val = x / 1000
        return f"{val:.1f}k".replace(".0k", "k")
    return f"{int(x)}"


fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(14, 13))


for col, (group_name, d) in enumerate(data.items()):
    x = d["dim"].values

    axes[0, col].plot(x, d["ari_score"], marker="o", linewidth=2)

    axes[1, col].plot(x, d["nmi_score"], marker="o", linewidth=2)

    axes[2, col].errorbar(
        x, d["isil_mean"], yerr=d["isil_std"], marker="o", capsize=3, linewidth=2
    )

    axes[3, col].errorbar(
        x, d["csil_mean"], yerr=d["csil_std"], marker="o", capsize=3, linewidth=2
    )

    axes[4, col].errorbar(
        x, d["graphcc_mean"], yerr=d["graphcc_std"], marker="o", capsize=3, linewidth=2
    )

    axes[0, col].set_title(group_name, fontsize=12, fontweight="bold")

    for row in range(5):
        ax = axes[row, col]
        ax.set_xscale("log")
        ax.set_xticks(x)

        if col in [0, 3]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k))
        else:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="x")

        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis="x", rotation=45)


        if row == 4:
            label = "Bins" if col == 0 else "Dimension"
            label = "Bins" if col == 3 else "Dimension"
            ax.set_xlabel(label, fontsize=11, fontweight="bold")

        ax.grid(True, which="major", alpha=0.3, linestyle="--")

y_labels = ["ARI", "NMI", "iSIL", "cSIL", "Graph connectivity"]
for row, label in enumerate(y_labels):
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")

for row in range(5):
    for col in range(4):
        axes[row, col].set_ylim(0, 1.05)


plt.tight_layout()
plt.savefig(OUTFILE_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTFILE_PDF, dpi=300, bbox_inches="tight")
