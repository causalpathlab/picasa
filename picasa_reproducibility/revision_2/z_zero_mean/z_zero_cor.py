import os
import sys
sys.path.append("/Users/sishirsubedi/Documents/projects/picasa/")
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from picasa import model
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


def decode_latent(z_comm):
    h = picasa_unique_model.u_decoder(torch.cat((z_comm, z_unique), dim=1))
    scale = torch.exp(picasa_unique_model.zinb_scale(h))
    recons = get_zinb_reconstruction(
        scale, 
        picasa_unique_model.zinb_dispersion.exp(), 
        picasa_unique_model.zinb_dropout(h)
    )
    return torch.log1p(recons)
            
all_records = []

for sample in samples:
    print(f"\n================ Running Baseline Comparison for {sample} ================")
    wdir = os.path.join(pp, sample)
    
    picasa_adata = ad.read_h5ad(os.path.join(wdir, "results", "picasa.h5ad"))
    adata = ad.read_h5ad(os.path.join(wdir, "data", f"all_{sample}.h5ad"))

    num_batches = len(picasa_adata.obs["batch"].unique())
    input_dim = adata.shape[1]
    nn_params = picasa_adata.uns["nn_params"]
    enc_layers = [128, 15]
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
        df_common.index = ["@".join(x.split("@")[:2]) for x in picasa_adata.obs.index.values]
    else:
        df_common.index = [x.split("@")[0] for x in picasa_adata.obs.index.values]


    # Global mean across all cells and all patients in this tissue
    x_zcommon_all = torch.tensor(
        df_common.loc[adata.obs_names].values,
        dtype=torch.float32,
    )

    x_zcommon_global_mean = x_zcommon_all.mean(dim=0)


    with torch.no_grad():
        for p1 in adata.obs["batch"].unique():
            print(f"[{sample}] Processing Batch: {p1}")

            current_adata = adata[adata.obs["batch"] == p1].copy()
            df = current_adata.to_df()

            x_c1 = torch.tensor(df.values).float()
            df_z = df_common.loc[df.index.values]
            x_zcommon = torch.tensor(df_z.values).float()
            
            x_zcommon_zeros = torch.zeros_like(x_zcommon)                      
            x_zcommon_mean = x_zcommon_global_mean.unsqueeze(0).expand(
                x_zcommon.shape[0], -1
            )             
            
            z_unique = picasa_unique_model.u_encoder(x_c1)
            z_unique_zeros = torch.zeros_like(z_unique)


            target_log = torch.log1p(x_c1)
            full_log = decode_latent(x_zcommon)
            zeros_log = decode_latent(x_zcommon_zeros)
            mean_log = decode_latent(x_zcommon_mean)


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
            common_log = torch.log1p(common_recons)
            
            
            mse_full = F.mse_loss(target_log, full_log).item()
            mse_zeros = F.mse_loss(target_log, zeros_log).item()
            mse_mean = F.mse_loss(target_log, mean_log).item()
            mse_common = F.mse_loss(target_log, common_log).item()
            

            corr_full = np.corrcoef(target_log.numpy().flatten(), full_log.numpy().flatten())[0, 1]
            corr_zeros = np.corrcoef(target_log.numpy().flatten(), zeros_log.numpy().flatten())[0, 1]
            corr_mean = np.corrcoef(target_log.numpy().flatten(), mean_log.numpy().flatten())[0, 1]
            corr_common = np.corrcoef(target_log.numpy().flatten(), common_log.numpy().flatten())[0, 1]

            all_records.append({
                "sample": sample,
                "tissue": sample_metadata[sample],
                "batch": p1,
                "MSE_Full": mse_full,
                "MSE_Common": mse_common,
                "MSE_Zeros": mse_zeros,
                "MSE_Mean": mse_mean,
                "Corr_Full": corr_full,
                "Corr_Common": corr_common,
                "Corr_Zeros": corr_zeros,
                "Corr_Mean": corr_mean
            })
                        
df_all = pd.DataFrame(all_records)
df_all.to_csv("baseline_comparison_records.csv", index=False)


