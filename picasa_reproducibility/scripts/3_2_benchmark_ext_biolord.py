import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc
import biolord

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1] 
WDIR = sys.argv[2]

DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
PATTERN = f'{SAMPLE}_*.h5ad'


def load_batches(data_dir, pattern, max_batches=25):
    batch_files = glob.glob(os.path.join(data_dir, pattern))
    batch_map = {}
    for i, file in enumerate(batch_files):
        if i >= max_batches:
            break
        batch_name = os.path.basename(file).replace('.h5ad', '').replace(f'{SAMPLE}_', '')
        logging.info(f"Loading {batch_name}")
        batch_map[batch_name] = an.read_h5ad(file)
    return batch_map


    

batch_map = load_batches(DATA_DIR, PATTERN)

adata_combined = sc.concat(batch_map, join="outer", label="batch")


biolord.Biolord.setup_anndata(
    adata_combined,
    ordered_attributes_keys=None,
    categorical_attributes_keys=['batch','celltype'],
)


module_params = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "seed": 42,
}
model = biolord.Biolord(
    adata=adata_combined,
    n_latent=15,
    model_name="simulated_batch",
    module_params=module_params,
    train_classifiers=False,
)


trainer_params = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-4,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}


model.train(
    max_epochs=400,
    batch_size=512,
    plan_kwargs=trainer_params,
    early_stopping=True,
    early_stopping_patience=20,
    check_val_every_n_epoch=10,
    num_workers=1,
    enable_checkpointing=False,
)


adata_preps = model.get_latent_representation_adata()[1]


adata_preps.obsm['unknown']= adata_preps.X[:,:15]
adata_preps.obsm['batch']= adata_preps.X[:,15:19]
adata_preps.obsm['celltype']= adata_preps.X[:,19:]

sc.pp.neighbors(adata_preps, use_rep="celltype")
sc.tl.umap(adata_preps)
sc.pl.umap(adata_preps, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.BATCH+'.png'))
plt.close()

sc.pl.umap(adata_preps, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.GROUP+'.png'))
plt.close()


pd.DataFrame(adata_preps.obsm['celltype'],index=adata_combined.obs.index.values).to_csv(os.path.join(RESULTS_DIR, 'benchmark_biolord.csv.gz'),compression='gzip')


sc.pp.neighbors(adata_preps, use_rep="batch")
sc.tl.umap(adata_preps)
sc.pl.umap(adata_preps, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.BATCH+'_unique.png'))
plt.close()

sc.pl.umap(adata_preps, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.GROUP+'_unique.png'))
plt.close()

sc.pp.neighbors(adata_preps, use_rep="unknown")
sc.tl.umap(adata_preps)
sc.pl.umap(adata_preps, color=[constants.BATCH],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.BATCH+'_unknown.png'))
plt.close()

sc.pl.umap(adata_preps, color=[constants.GROUP],legend_loc=None)
plt.savefig(os.path.join(RESULTS_DIR, 'scanpy_biolord_umap_'+constants.GROUP+'_unknown.png'))
plt.close()