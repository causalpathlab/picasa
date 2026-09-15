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
# samples = ["brca", "lung", "ovary"]
samples = ["ovary"]

sample_metadata = {
    "brca": "Breast (Tumour)",
    "lung": "Lung (Tumour)",
    "ovary": "Ovary (Tumour)",
}


def get_zinb_reconstruction(px_s, px_r, px_d):
    zinb_dist = ZeroInflatedNegativeBinomial(
        mu=px_s, theta=px_r, zi_logits=px_d
    )
    return zinb_dist.mean

all_records = []

for sample in samples:
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

    # Load model results & full adata
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

    target_patient = []
    full_patient = []
    common_patient = []
    unique_patient = []
    cell_names = []


    ct = "EOC"  
    # ct == "Malignant"

    with torch.no_grad():
        for p1 in adata.obs["batch"].unique():
            current_adata = adata[
                (adata.obs["batch"] == p1)
                & (adata.obs["celltype"] == ct)
            ].copy()

            if current_adata.shape[0] == 0:
                continue

            df = current_adata.to_df()
            print(f"Processing Batch: {p1} | Celltype: {ct} | Shape: {df.shape}")

            x_c1 = torch.tensor(df.values).float()
            df_z = df_common.loc[df.index.values]
            x_zcommon = torch.tensor(df_z.values).float()

            z_unique = picasa_unique_model.u_encoder(x_c1)
            z_unique_zeros = torch.zeros_like(z_unique)
            x_zcommon_zeros = torch.zeros_like(x_zcommon)

            full_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon, z_unique), dim=1)
            )
            full_recons = get_zinb_reconstruction(
                torch.exp(picasa_unique_model.zinb_scale(full_h)),
                picasa_unique_model.zinb_dispersion.exp(),
                picasa_unique_model.zinb_dropout(full_h),
            )

            unique_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon_zeros, z_unique), dim=1)
            )
            unique_recons = get_zinb_reconstruction(
                torch.exp(picasa_unique_model.zinb_scale(unique_h)),
                picasa_unique_model.zinb_dispersion.exp(),
                picasa_unique_model.zinb_dropout(unique_h),
            )

            common_h = picasa_unique_model.u_decoder(
                torch.cat((x_zcommon, z_unique_zeros), dim=1)
            )
            common_recons = get_zinb_reconstruction(
                torch.exp(picasa_unique_model.zinb_scale(common_h)),
                picasa_unique_model.zinb_dispersion.exp(),
                picasa_unique_model.zinb_dropout(common_h),
            )

            target_patient.append(x_c1)
            full_patient.append(full_recons)
            common_patient.append(common_recons)
            unique_patient.append(unique_recons)
            cell_names.extend(df.index.tolist())


    all_target = torch.cat(target_patient, dim=0).cpu().numpy()
    all_full = torch.cat(full_patient, dim=0).cpu().numpy()
    all_common = torch.cat(common_patient, dim=0).cpu().numpy()
    all_unique = torch.cat(unique_patient, dim=0).cpu().numpy()

    
    gene_names = adata.var_names.values

    pd.DataFrame(all_target, index=cell_names, columns=gene_names).to_parquet("data/raw_recons.parquet")
    pd.DataFrame(all_full, index=cell_names, columns=gene_names).to_parquet("data/full_recons.parquet")
    pd.DataFrame(all_common, index=cell_names, columns=gene_names).to_parquet("data/common_recons.parquet")
    pd.DataFrame(all_unique, index=cell_names, columns=gene_names).to_parquet("data/unique_recons.parquet")
