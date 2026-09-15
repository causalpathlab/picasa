import os

import sys 
sys.path.append('/Users/sishirsubedi/Documents/projects/picasa/')

import glob
import logging
import matplotlib.pyplot as plt
import anndata as an
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import picasa 
from scipy.stats import pearsonr
import torch.nn.functional as F

from picasa import model


sample = 'ovary'

common_epochs = 1
common_meta_epoch = 15
unique_epoch = 250
base_epoch = 250


input_dim_map = {
	'brca':2037,
	'lung':2020,
	'ovary':2000
}

params = {'device' : 'cpu',
		'batch_size' : 100,
		'input_dim' : input_dim_map[sample],
		'embedding_dim' : 3000,
		'attention_dim' : 25,
		'latent_dim' : 25,
		'encoder_layers' : [100,25],
		'projection_layers' : [50,50],
		'learning_rate' : 1e-5,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 0.75,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': common_epochs,
		'meta_epochs': common_meta_epoch
		}   
  
picasa_model = model.PICASACommonNet(params['input_dim'], params['embedding_dim'],params['attention_dim'], params['latent_dim'], params['encoder_layers'], params['projection_layers'],params['corruption_tol'],params['pair_importance_weight']).to(params['device'])


wdir = '/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/'
ddir = wdir+sample+'/data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=25:
		break

picasa_object = picasa.create_picasa_object(
	batch_map,
    sample,
	wdir+sample
 	)



picasa_model.load_state_dict(torch.load(wdir+sample+'/results/picasa_common.model', map_location=torch.device(params['device'])))
picasa_model.eval()
picasa_result = sc.read_h5ad(wdir+sample+'/results/picasa.h5ad')


adata_combined = an.read_h5ad(wdir+sample+'/data/all_'+sample+'.h5ad')
input_dim = adata_combined.X.shape[1]
enc_layers = [128,25]
unique_latent_dim = 25
common_latent_dim = picasa_result.obsm['common'].shape[1]
dec_layers = [128,128]


num_batches = len(picasa_result.obs['batch'].unique())
picasa_unique_model = model.PICASAUniqueNet(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(params['device'])
picasa_unique_model.load_state_dict(torch.load(wdir+sample+'/results/picasa_unique.model', map_location=torch.device(params['device'])))

picasa_unique_model.eval()

from scvi.distributions import ZeroInflatedNegativeBinomial

def get_zinb_reconstruction(px_s, px_r, px_d):
    zi_prob = torch.sigmoid(px_d)
    return (1 - zi_prob) * px_s


records = []
with torch.no_grad():
		
	for r in picasa_result.uns['nbr_map'].itertuples():
		
		source = r.batch_pair.split('_')[0]
		nbr = r.batch_pair.split('_')[1]
		source_index = r.key
		nbr_index = r.neighbor
		
		source_x = picasa_object.data.adata_list[source].X[source_index]
		source_x_name = picasa_object.data.adata_list[source].obs.index[source_index]+'@'+source
		source_zc = picasa_result.obsm['common'].loc[source_x_name].values
  		
		nbr_x = picasa_object.data.adata_list[nbr].X[nbr_index]
		nbr_x_name = picasa_object.data.adata_list[nbr].obs.index[nbr_index]+'@'+nbr
		nbr_zc = picasa_result.obsm['common'].loc[nbr_x_name].values
		
  
		n_cells = picasa_object.data.adata_list[nbr].n_obs
		candidates = np.setdiff1d(
 				     np.arange(n_cells),
    				 [nbr_index]
					)
		random_index = np.random.choice(candidates)
		random_x = picasa_object.data.adata_list[nbr].X[random_index]
		random_x_name = picasa_object.data.adata_list[nbr].obs.index[random_index]+'@'+nbr
		random_zc = picasa_result.obsm['common'].loc[random_x_name].values		
  
  
		if sample =='lung':
			source_x_raw_index = adata_combined.obs_names.get_loc('@'.join(source_x_name.split('@')[:2]))
		else:
			source_x_raw_index = adata_combined.obs_names.get_loc(source_x_name.split('@')[0])
		source_x_raw = adata_combined.X[source_x_raw_index]
		source_x_raw_exp = np.expm1(source_x_raw.toarray())
		x_c1 = torch.tensor(source_x_raw_exp).float()
		x_z = torch.tensor(source_zc).float().unsqueeze(0)
		z = picasa_unique_model(x_c1,x_z)
		# z_u = z[0]
		px_scale = z[1]
		px_rate = z[2]
		px_dropout = z[3]
		batch_pred = z[4]
		source_x_recons = get_zinb_reconstruction(px_scale,px_rate,px_dropout)
		

		x_zn = torch.tensor(nbr_zc).float().unsqueeze(0)
		zn = picasa_unique_model(x_c1,x_zn)
		# z_un = zn[0]
		px_scalen = zn[1]
		px_raten = zn[2]
		px_dropoutn = zn[3]
		batch_predn = zn[4]
		nbr_x_recons = get_zinb_reconstruction(px_scalen,px_raten,px_dropoutn)
		
		
		x_zr = torch.tensor(random_zc).float().unsqueeze(0)
		zr = picasa_unique_model(x_c1,x_zr)
		# z_ur = zr[0]
		px_scaler = zr[1]
		px_rater = zr[2]
		px_dropoutr = zr[3]
		batch_predr = zr[4]
		random_x_recons = get_zinb_reconstruction(px_scaler,px_rater,px_dropoutr)
		

		recon_orig = torch.log1p(source_x_recons).detach().cpu().numpy().flatten()
		recon_swapped = torch.log1p(nbr_x_recons).detach().cpu().numpy().flatten()
		recon_random = torch.log1p(random_x_recons).detach().cpu().numpy().flatten()
  
		
		zc_on_corr, _ = pearsonr(source_zc, nbr_zc)
		zc_or_corr, _ = pearsonr(source_zc, random_zc)
    
		mse_paired = F.mse_loss(
			source_x_recons,
			nbr_x_recons
		).item()

		mse_random = F.mse_loss(
			source_x_recons,
			random_x_recons
		).item()
  
		records.append(
                {
                    "zc_on_corr": zc_on_corr,
                    "zc_or_corr": zc_or_corr,
                    "mse_paired": mse_paired,  
                    "mse_random": mse_random, 
                } )

df = pd.DataFrame(records)
df.to_csv(sample+'_recons_test.csv')



######plot

import matplotlib.gridspec as gridspec
import seaborn as sns

samples = ["breast", "lung", "ovary"]

sam_map={
	"breast":"Breast",
	"lung":"Lung",
	"ovary":"Ovary"
}
colors = ["#2b5c8f", "#d95f02"]

# Set global style
sns.set_theme(style="whitegrid", font_scale=1.0)

fig = plt.figure(figsize=(10, 12))
outer_gs = gridspec.GridSpec(3, 1, figure=fig, wspace=0.3)

for i, sample in enumerate(samples):
    df = pd.read_csv(f"{sample}_recons_test.csv", index_col=0)
    df = df[df['mse_random']<5]
    df = df[df['mse_paired']<5]

    print(f"=== {sample.upper()} ===")
    print(df[["zc_on_corr", "zc_or_corr", "mse_paired", "mse_random"]].describe())
    print("\nMedians:")
    print(df[["zc_on_corr", "zc_or_corr", "mse_paired", "mse_random"]].median())
    print("-" * 50)

    inner_gs = outer_gs[i].subgridspec(
        2, 2, width_ratios=[1, 1.1], height_ratios=[1, 3], hspace=0.12, wspace=0.35
    )

    ax_corr = fig.add_subplot(inner_gs[:, 0])    
    ax_mse_top = fig.add_subplot(inner_gs[0, 1]) 
    ax_mse_bot = fig.add_subplot(inner_gs[1, 1]) 


    df_corr = pd.DataFrame(
        {"Paired": df["zc_on_corr"], "Random": df["zc_or_corr"]}
    ).melt(var_name="Group", value_name="Correlation")

    sns.boxplot(
        data=df_corr,
        x="Group",
        y="Correlation",
        palette=colors,
        width=0.4,
        fliersize=2,
        ax=ax_corr,
    )

    ax_corr.set_title(f"{sam_map[sample]}: Z-Correlation", fontweight="bold", fontsize=11)
    ax_corr.set_xlabel("")
    ax_corr.set_ylabel("Pearson ($r$)", fontweight="bold")


    df_mse = pd.DataFrame(
        {"Paired": df["mse_paired"], "Random": df["mse_random"]}
    ).melt(var_name="Group", value_name="MSE")

    for ax in [ax_mse_top, ax_mse_bot]:
        sns.boxplot(
            data=df_mse,
            x="Group",
            y="MSE",
            palette=colors,
            width=0.4,
            fliersize=2,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    mse_q1 = df_mse["MSE"].quantile(0.25)
    mse_q3 = df_mse["MSE"].quantile(0.75)
    mse_iqr = mse_q3 - mse_q1
    max_regular_val = mse_q3 + 2.5 * mse_iqr

    ax_mse_bot.set_ylim(0, max_regular_val)
    ax_mse_top.set_ylim(df_mse["MSE"].max() * 0.85, df_mse["MSE"].max() * 1.05)

    ax_mse_top.spines["bottom"].set_visible(False)
    ax_mse_bot.spines["top"].set_visible(False)
    ax_mse_top.xaxis.tick_top()
    ax_mse_top.tick_params(labeltop=False, top=False, bottom=False)
    ax_mse_bot.xaxis.tick_bottom()

    ax_mse_top.set_title(f"{sam_map[sample]}: MSE", fontweight="bold", fontsize=11)
    ax_mse_top.set_ylabel("")
    ax_mse_bot.set_ylabel("Mean Squared Error (MSE)", fontweight="bold")

    d = 0.025
    kwargs = dict(transform=ax_mse_top.transAxes, color="k", clip_on=False)
    ax_mse_top.plot((-d, +d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_mse_bot.transAxes)
    ax_mse_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)

output_path_png = "picasa_recons.png"
output_path_pdf = "picasa_recons.pdf"
plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
plt.savefig(output_path_pdf, bbox_inches="tight")




results = []
for i, sample in enumerate(samples):
    df = pd.read_csv(f"{sample}_recons_test.csv", index_col=0)
    df = df[(df["mse_random"] < 5) & (df["mse_paired"] < 5)]
    print(df.describe())
    med_r = df["zc_on_corr"].median()
    std_r = df["zc_on_corr"].std()
    label = sam_map.get(sample, sample.capitalize())
    results.append(f"{med_r:.3f} \u00b1 {std_r:.3f} ({label.lower()})")
print("median r = " + ", ".join(results))



for i, sample in enumerate(samples):
    df = pd.read_csv(f"{sample}_recons_test.csv", index_col=0)
    df = df[(df["mse_random"] < 5) & (df["mse_paired"] < 5)]

    paired_median = df["mse_paired"].median()
    paired_75th = df["mse_paired"].quantile(0.75)

    random_median = df["mse_random"].median()
    random_75th = df["mse_random"].quantile(0.75)

    print(f"=== {sample.upper()} MSE SUMMARY ===")
    print(
        f"Paired MSE: Median = {paired_median:.4f} | 75th Pct = {paired_75th:.4f}"
    )
    print(
        f"Random MSE: Median = {random_median:.4f} | 75th Pct = {random_75th:.4f}\n"
    )
    
    
    
#     === BREAST MSE SUMMARY ===
# Paired MSE: Median = 0.0089 | 75th Pct = 0.0229
# Random MSE: Median = 0.2729 | 75th Pct = 0.4330

# === LUNG MSE SUMMARY ===
# Paired MSE: Median = 0.0024 | 75th Pct = 0.0063
# Random MSE: Median = 0.0217 | 75th Pct = 0.2060

# === OVARY MSE SUMMARY ===
# Paired MSE: Median = 0.0227 | 75th Pct = 0.0621
# Random MSE: Median = 0.9233 | 75th Pct = 2.4722