import glob
import os
import sys

sys.path.append("/Users/sishirsubedi/Documents/projects/picasa/")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from picasa import dutil, model
from scvi.distributions import ZeroInflatedNegativeBinomial

pp = "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/"
samples = ["brca", "lung", "ovary"]

sample_metadata = {
    "brca": "Breast (Tumour)",
    "lung": "Lung (Tumour)",
    "ovary": "Ovary (Tumour)",
}


def get_zinb_reconstruction(px_s, px_r, px_d):
    zinb_dist = ZeroInflatedNegativeBinomial(
        mu=px_s, theta=px_r, zi_logits=px_d
    )
    return zinb_dist.sample()


all_records = []

for sample in samples:
    print(f"\n================ Running Ablation for {sample} ================")
    ddir = os.path.join(pp, sample, "data")
    pattern = sample + "_*.h5ad"

    file_paths = glob.glob(os.path.join(ddir, pattern))
    file_names = [os.path.basename(file_path) for file_path in file_paths]

    batch_map = {}
    batch_count = 0
    for file_name in file_names:
        key = file_name.replace(".h5ad", "").replace(sample + "_", "")
        batch_map[key] = ad.read_h5ad(os.path.join(ddir, file_name))
        batch_count += 1
        if batch_count >= 25:
            break

    wdir = os.path.join(pp, sample)
    picasa_adata = ad.read_h5ad(os.path.join(wdir, "results", "picasa.h5ad"))
    adata = ad.read_h5ad(os.path.join(wdir, "data", f"all_{sample}.h5ad"))

    num_batches = len(picasa_adata.obs["batch"].unique())
    input_dim = adata.shape[1]
    nn_params = picasa_adata.uns["nn_params"]
    enc_layers = [128, 25]
    unique_latent_dim = nn_params["latent_dim"]
    common_latent_dim = nn_params["latent_dim"]
    dec_layers = [128, 128]
    nn_params["device"] = "cpu"

    picasa_unique_model = model.PICASAUniqueNet(
        input_dim,
        common_latent_dim,
        unique_latent_dim,
        enc_layers,
        dec_layers,
        num_batches,
    ).to(nn_params["device"])

    picasa_unique_model.load_state_dict(
        torch.load(
            os.path.join(wdir, "results", "picasa_unique.model"),
            map_location=torch.device(nn_params["device"]),
        )
    )
    picasa_unique_model.eval()

    df_common = picasa_adata.obsm["common"]
    if sample == "lung":
        df_common.index = [
            "@".join(x.split("@")[:2]) for x in picasa_adata.obs.index.values
        ]
    else:
        df_common.index = [
            x.split("@")[0] for x in picasa_adata.obs.index.values
        ]

    with torch.no_grad():
        for p1 in adata.obs["batch"].unique():
            print(f"[{sample}] Processing Batch: {p1}")

            current_adata = adata[adata.obs["batch"] == p1].copy()
            df = current_adata.to_df()

            x_c1 = torch.tensor(df.values).float()
            df_z = df_common.loc[df.index.values]
            x_zcommon = torch.tensor(df_z.values).float()

            z_unique = picasa_unique_model.u_encoder(x_c1)
            z_unique_zeros = torch.zeros_like(z_unique)
            x_zcommon_zeros = torch.zeros_like(x_zcommon)


            full_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon, z_unique), dim=1)
            )
            full_px_scale = torch.exp(picasa_unique_model.zinb_scale(full_h))
            full_px_dropout = picasa_unique_model.zinb_dropout(full_h)
            full_px_rate = picasa_unique_model.zinb_dispersion.exp()
            full_recons = get_zinb_reconstruction(
                full_px_scale, full_px_rate, full_px_dropout
            )

            # Unique Only
            unique_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon_zeros, z_unique), dim=1)
            )
            unique_px_scale = torch.exp(
                picasa_unique_model.zinb_scale(unique_h)
            )
            unique_px_dropout = picasa_unique_model.zinb_dropout(unique_h)
            unique_px_rate = picasa_unique_model.zinb_dispersion.exp()
            unique_recons = get_zinb_reconstruction(
                unique_px_scale, unique_px_rate, unique_px_dropout
            )

            # Common Only
            common_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon, z_unique_zeros), dim=1)
            )
            common_px_scale = torch.exp(
                picasa_unique_model.zinb_scale(common_h)
            )
            common_px_dropout = picasa_unique_model.zinb_dropout(common_h)
            common_px_rate = picasa_unique_model.zinb_dispersion.exp()
            common_recons = get_zinb_reconstruction(
                common_px_scale, common_px_rate, common_px_dropout
            )

            target_log = torch.log1p(x_c1)
            full_log = torch.log1p(full_recons)
            common_log = torch.log1p(common_recons)
            unique_log = torch.log1p(unique_recons)

            mse_full = F.mse_loss(target_log, full_log).item()
            mse_common = F.mse_loss(target_log, common_log).item()
            mse_unique = F.mse_loss(target_log, unique_log).item()

            corr_full = np.corrcoef(
                target_log.numpy().flatten(), full_log.numpy().flatten()
            )[0, 1]
            corr_common = np.corrcoef(
                target_log.numpy().flatten(), common_log.numpy().flatten()
            )[0, 1]
            corr_unique = np.corrcoef(
                target_log.numpy().flatten(), unique_log.numpy().flatten()
            )[0, 1]

            all_records.append(
                {
                    "sample": sample,
                    "tissue": sample_metadata[sample],
                    "batch": p1,
                    "MSE_Full": mse_full,
                    "MSE_Common": mse_common,
                    "MSE_Unique": mse_unique,
                    "Corr_Full": corr_full,
                    "Corr_Common": corr_common,
                    "Corr_Unique": corr_unique,
                }
            )

df_all = pd.DataFrame(all_records)

df_all.to_csv("all_records.csv",index=False)


###### plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

samples_order = ["brca", "lung", "ovary"]
fig, axes = plt.subplots(3, 2, figsize=(11, 10), dpi=300, sharex=True)

palette = {
    "Full Model": "#2b5c8f",
    "Common Only": "#4a86e8",
    "Unique Only": "#d97724",
}

for i, sample in enumerate(samples_order):
    df_sample = df_all[df_all["sample"] == sample]
    tissue_label = df_sample["tissue"].iloc[0]

    df_mse = df_sample[
        ["batch", "MSE_Full", "MSE_Common", "MSE_Unique"]
    ].melt(id_vars=["batch"], var_name="Condition", value_name="MSE")
    df_mse["Condition"] = df_mse["Condition"].map(
        {
            "MSE_Full": "Full Model",
            "MSE_Common": "Common Only",
            "MSE_Unique": "Unique Only",
        }
    )

    ax_mse = axes[i, 0]
    sns.barplot(
        data=df_mse,
        x="Condition",
        y="MSE",
        hue="Condition",
        ax=ax_mse,
        palette=palette,
        capsize=0.1,
        edgecolor="black",
        linewidth=0.8,
    )

    sns.stripplot(
        data=df_mse,
        x="Condition",
        y="MSE",
        ax=ax_mse,
        color="black",
        alpha=0.6,
        jitter=0.15,
        size=4,
    )

    ax_mse.set_title(
        f"{tissue_label} — Reconstruction Error (MSE)",
        fontweight="bold",
        fontsize=11,
    )
    ax_mse.set_ylabel("MSE", fontweight="bold")
    ax_mse.set_xlabel("")

    df_corr = df_sample[
        ["batch", "Corr_Full", "Corr_Common", "Corr_Unique"]
    ].melt(id_vars=["batch"], var_name="Condition", value_name="Correlation")
    df_corr["Condition"] = df_corr["Condition"].map(
        {
            "Corr_Full": "Full Model",
            "Corr_Common": "Common Only",
            "Corr_Unique": "Unique Only",
        }
    )

    ax_corr = axes[i, 1]
    sns.barplot(
        data=df_corr,
        x="Condition",
        y="Correlation",
        hue="Condition",
        ax=ax_corr,
        palette=palette,
        capsize=0.1,
        edgecolor="black",
        linewidth=0.8,
    )

    sns.stripplot(
        data=df_corr,
        x="Condition",
        y="Correlation",
        ax=ax_corr,
        color="black",
        alpha=0.6,
        jitter=0.15,
        size=4,
    )

    ax_corr.set_title(
        f"{tissue_label} — Expression Fidelity (Pearson r)",
        fontweight="bold",
        fontsize=11,
    )
    ax_corr.set_ylabel("Pearson Correlation (r)", fontweight="bold")
    ax_corr.set_xlabel("")

    if i == 2:
        ax_mse.tick_params(axis="x", rotation=15)
        ax_corr.tick_params(axis="x", rotation=15)

sns.despine()
plt.tight_layout()
plt.savefig("Ablation_By_Tissue_3x2.png", dpi=300, bbox_inches="tight")
plt.savefig("Ablation_By_Tissue_3x2.pdf", bbox_inches="tight")



samples_order = ["brca", "lung", "ovary"]
metrics = ["MSE", "Corr"]
conditions = ["Full", "Common", "Unique"]
tissue_map = {"brca": "breast", "lung": "lung", "ovary": "ovary"}

for metric in metrics:
    print(f"=== {metric} METRICS ===")
    for cond in conditions:
        col_name = f"{metric}_{cond}"
        results = []

        for sample in samples_order:
            df_sample = df_all[df_all["sample"] == sample]

            med_val = df_sample[col_name].mean()
            std_val = df_sample[col_name].std()

            label = tissue_map.get(sample, sample)
            results.append(f"{med_val:.3f} \u00b1 {std_val:.3f} ({label})")

        print(f"{metric} ({cond}): median = " + ", ".join(results))
    print("\n" + "-" * 50)
    
# === MSE METRICS ===
# MSE (Full): median = 0.110 ± 0.004 (breast), 0.092 ± 0.004 (lung), 0.076 ± 0.003 (ovary)
# MSE (Common): median = 0.154 ± 0.005 (breast), 0.148 ± 0.005 (lung), 0.078 ± 0.004 (ovary)
# MSE (Unique): median = 0.124 ± 0.009 (breast), 0.095 ± 0.004 (lung), 0.084 ± 0.005 (ovary)

# --------------------------------------------------
# === Corr METRICS ===
# Corr (Full): median = 0.402 ± 0.040 (breast), 0.399 ± 0.031 (lung), 0.383 ± 0.029 (ovary)
# Corr (Common): median = 0.329 ± 0.027 (breast), 0.306 ± 0.033 (lung), 0.388 ± 0.033 (ovary)
# Corr (Unique): median = 0.352 ± 0.030 (breast), 0.380 ± 0.042 (lung), 0.302 ± 0.011 (ovary)

# --------------------------------------------------