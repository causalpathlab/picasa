import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42

df_all = pd.read_csv("baseline_comparison_records.csv")
samples_order = ["brca", "lung", "ovary"]

order = [
    "Full Model (Z+Y)",
    "Common (Z,$Y=0$)",
    "Unique Zeros ($Z=0$,Y)",
    "Unique Global Mean ($Z=\\mu$,Y)",
]

palette = {
    "Full Model (Z+Y)": "#1f3f61",
    "Common (Z,$Y=0$)": "#3c67ab",
    "Unique Zeros ($Z=0$,Y)": "#a65d1e",
    "Unique Global Mean ($Z=\\mu$,Y)": "#3e7b4c",
}

BAR_WIDTH = 0.80
POINT_SIZE = 3.5
POINT_ALPHA = 0.6

fig, axes = plt.subplots(
    3, 2,
    figsize=(14, 12),
    dpi=300,
    sharex=False,
)

for i, sample in enumerate(samples_order):
    df_sample = df_all[df_all["sample"] == sample]

    ax_mse = axes[i, 0]
    ax_corr = axes[i, 1]

    if df_sample.empty:
        ax_mse.set_visible(False)
        ax_corr.set_visible(False)
        continue

    tissue_label = (
        df_sample["tissue"].iloc[0]
        if "tissue" in df_sample.columns
        else sample
    )

    df_mse = df_sample[
        ["batch", "MSE_Full", "MSE_Common", "MSE_Zeros", "MSE_Mean"]
    ].melt(
        id_vars=["batch"],
        var_name="Condition",
        value_name="MSE",
    )

    df_mse["Condition"] = df_mse["Condition"].map({
        "MSE_Full": "Full Model (Z+Y)",
        "MSE_Common": "Common (Z,$Y=0$)",
        "MSE_Zeros": "Unique Zeros ($Z=0$,Y)",
        "MSE_Mean": "Unique Global Mean ($Z=\\mu$,Y)",
    })

    means_mse = (
        df_mse.groupby("Condition")["MSE"]
        .mean()
        .reindex(order)
    )

    ax_mse.bar(
        range(len(order)),
        means_mse.values,
        width=BAR_WIDTH,
        color=[palette[c] for c in order],
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )

    for j, condition in enumerate(order):
        values = df_mse.loc[
            df_mse["Condition"] == condition, "MSE"
        ].dropna().values

        if len(values):
            offsets = [
                0.0
                if len(values) == 1
                else (k / (len(values) - 1) - 0.5) * 0.12
                for k in range(len(values))
            ]

            ax_mse.scatter(
                [j + offset for offset in offsets],
                values,
                color="black",
                s=POINT_SIZE ** 2,
                alpha=POINT_ALPHA,
                zorder=4,
            )

    ax_mse.set_xticks(range(len(order)))
    ax_mse.set_xticklabels(order)
    
    ax_mse.tick_params(axis="x", which="both", bottom=True, labelbottom=True, rotation=15)
    ax_mse.set_title(
        f"{tissue_label} — Reconstruction Error (MSE)",
        fontweight="bold",
        fontsize=11,
    )
    ax_mse.set_ylabel("MSE", fontweight="bold")
    # ax_mse.set_xlabel("Condition", fontweight="bold")
    ax_mse.set_axisbelow(True)

    # Correlation
    df_corr = df_sample[
        ["batch", "Corr_Full", "Corr_Common", "Corr_Zeros", "Corr_Mean"]
    ].melt(
        id_vars=["batch"],
        var_name="Condition",
        value_name="Correlation",
    )

    df_corr["Condition"] = df_corr["Condition"].map({
        "Corr_Full": "Full Model (Z+Y)",
        "Corr_Common": "Common (Z,$Y=0$)",
        "Corr_Zeros": "Unique Zeros ($Z=0$,Y)",
        "Corr_Mean": "Unique Global Mean ($Z=\\mu$,Y)",
    })

    means_corr = (
        df_corr.groupby("Condition")["Correlation"]
        .mean()
        .reindex(order)
    )

    ax_corr.bar(
        range(len(order)),
        means_corr.values,
        width=BAR_WIDTH,
        color=[palette[c] for c in order],
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )

    for j, condition in enumerate(order):
        values = df_corr.loc[
            df_corr["Condition"] == condition, "Correlation"
        ].dropna().values

        if len(values):
            offsets = [
                0.0
                if len(values) == 1
                else (k / (len(values) - 1) - 0.5) * 0.12
                for k in range(len(values))
            ]

            ax_corr.scatter(
                [j + offset for offset in offsets],
                values,
                color="black",
                s=POINT_SIZE ** 2,
                alpha=POINT_ALPHA,
                zorder=4,
            )

    ax_corr.set_xticks(range(len(order)))
    ax_corr.set_xticklabels(order)
    
    ax_corr.tick_params(axis="x", which="both", bottom=True, labelbottom=True, rotation=15)
    ax_corr.set_title(
        f"{tissue_label} — Expression Fidelity (Pearson r)",
        fontweight="bold",
        fontsize=11,
    )
    ax_corr.set_ylabel("Pearson Correlation (r)", fontweight="bold")
    # ax_corr.set_xlabel("Condition", fontweight="bold")
    ax_corr.set_axisbelow(True)

sns.despine(top=True, right=True, left=False, bottom=False)
plt.tight_layout()
plt.savefig(
    "z_zero_test.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "z_zero_test.pdf",
    bbox_inches="tight",
)


#### Summarize 

import pandas as pd

df = pd.read_csv('baseline_comparison_records.csv')

metrics = [
    'MSE_Full', 'MSE_Common', 'MSE_Zeros', 'MSE_Mean', 
    'Corr_Full', 'Corr_Common', 'Corr_Zeros', 'Corr_Mean'
]

mean_df = df.groupby('tissue')[metrics].mean()
std_df = df.groupby('tissue')[metrics].std()

summary_df = pd.DataFrame()
for col in metrics:
    summary_df[col] = mean_df[col].round(4).astype(str) + " ± " + std_df[col].round(4).astype(str)

print(summary_df.to_string())

#                         MSE_Full       MSE_Common        MSE_Zeros         MSE_Mean        Corr_Full      Corr_Common       Corr_Zeros        Corr_Mean
# tissue                                                                                                                                                 
# Breast (Tumour)  0.1102 ± 0.0045  0.1553 ± 0.0051  0.1224 ± 0.0093  0.1138 ± 0.0087   0.403 ± 0.0396  0.3276 ± 0.0303  0.3629 ± 0.0342   0.374 ± 0.0406
# Lung (Tumour)    0.0925 ± 0.0043  0.1485 ± 0.0053  0.0952 ± 0.0044  0.0918 ± 0.0047  0.3972 ± 0.0311  0.3086 ± 0.0331  0.3727 ± 0.0418  0.3862 ± 0.0338
# Ovary (Tumour)   0.0762 ± 0.0035   0.0784 ± 0.004  0.0858 ± 0.0052  0.0817 ± 0.0038  0.3868 ± 0.0289   0.393 ± 0.0332  0.3018 ± 0.0113  0.3163 ± 0.0142
# >>> 