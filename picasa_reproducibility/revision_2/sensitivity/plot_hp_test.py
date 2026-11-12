import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns 

SAMPLE = "pancreas"
WDIR = (
    "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/"
    f"revision_2/sensitivity/{SAMPLE}"
)

SAMPLE_LIST_1 = [
    "results_nbr_all",
    "results_nbr_first",
    "results_nbr_last",
    "results_nbr_middle",
    "results_nbr_random",
]
df_list = []
for SAMPLE in SAMPLE_LIST_1:
  CWDIR = os.path.join(WDIR, SAMPLE)
  dfc = pd.read_csv(os.path.join(CWDIR, "benchmark_scores.csv.gz"))
  dfc["exp"] = SAMPLE.split("_")[2]
  df_list.append(dfc)

df = pd.concat(df_list, ignore_index=True)

OUTFILE_PNG = os.path.join(WDIR, "picasa_sensitivity_analysis.png")
OUTFILE_PDF = os.path.join(WDIR, "picasa_sensitivity_analysis.pdf")
os.makedirs(os.path.dirname(OUTFILE_PNG), exist_ok=True)

metrics_info = {
    "ARI": ("ari_score", None),
    "NMI": ("nmi_score", None),
    "cSIL": ("csil_mean", "csil_std"),
    "Graph Connectivity": ("graphcc_mean", "graphcc_std"),
}

legend_map = {
    "all": "Sequential (Closest Neighbour - Default Order)",
    "first": "All Paired (Closest Neighbour)",
    "last": "All Paired (Farthest Neighbour)",
    "middle": "All Paired (Median-Distance)",
    "random": "All Paired (Random)",
}

experiments = df["exp"].unique()

palette = sns.color_palette("deep", len(experiments))

labels = list(metrics_info.keys())
x = np.arange(len(labels))
width = 0.15  

fig, ax = plt.subplots(figsize=(10, 6))

for i, exp_val in enumerate(experiments):
  sub_df = df[df["exp"] == exp_val].iloc[0]

  means = []
  errs = []

  for label, (mean_col, std_col) in metrics_info.items():
    means.append(sub_df[mean_col])
    if std_col and std_col in df.columns:
      errs.append(sub_df[std_col])
    else:
      errs.append(0)

  offset = (i - len(experiments) / 2 + 0.5) * width
  display_label = legend_map.get(exp_val, exp_val)

  ax.bar(
      x + offset,
      means,
      width,
      yerr=errs,
      capsize=3,
      label=display_label,
      color=palette[i], 
      edgecolor="black",
      linewidth=0.5,
  )

ax.set_xlabel("Metrics", fontsize=12, labelpad=10)
ax.set_ylabel("Score Value", fontsize=12, labelpad=10)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15)

ax.yaxis.grid(True, which="major", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.legend(
    title="Cell Pairing Strategy",
    fontsize=10,
    title_fontsize=11,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.savefig(OUTFILE_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUTFILE_PDF, dpi=300, bbox_inches="tight")
plt.show()