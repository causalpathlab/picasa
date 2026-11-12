import sys
import glob
import os
import argparse
import anndata as an


sys.path.append('/Users/sishirsubedi/Documents/projects/picasa/')
import picasa

sample = "pancreas"
wdir = "/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/revision_2/sensitivity/pancreas/"

common_epochs = 1
common_meta_epoch = 15
unique_epoch = 250
base_epoch = 250


ddir = os.path.join(wdir, 'data')
pattern = f"{sample}_*.h5ad"

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
    key = file_name.replace('.h5ad', '').replace(f"{sample}_", '')
    batch_map[key] = an.read_h5ad(os.path.join(ddir, file_name))
    batch_count += 1
    if batch_count >= 12:
        break

out_dir = os.path.join(wdir)
os.makedirs(out_dir, exist_ok=True)
r_dir = os.path.join(wdir,"results")
os.makedirs(r_dir, exist_ok=True)

picasa_object = picasa.create_picasa_object(
    batch_map,
    sample,
    out_dir
)

params = {'device' : 'cpu',
		'batch_size' : 100,
		'input_dim' : 2000,
		'embedding_dim' : 3000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [25,25],
		'learning_rate' : 0.001,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 1.0,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': common_epochs,
		'meta_epochs': common_meta_epoch
		} 



picasa_object.estimate_neighbour(params['pair_search_method'])
picasa_object.set_nn_params(params)
picasa_object.train_common()
picasa_object.plot_loss(tag='common')

eval_batch_size = 500
picasa_object.eval_common(eval_batch_size, 'cpu')
picasa_object.save_model()

with open(os.path.join(out_dir, "completed.flag"), "w") as f:
    f.write("Done\n")