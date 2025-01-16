import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

import picasa
import anndata as an
import glob
import os


sample = sys.argv[1] 
wdir = sys.argv[2]
ed = sys.argv[3]
ad = sys.argv[4]
pl = sys.argv[5]
lr = sys.argv[6]
pw = sys.argv[7]
common_epochs = 1
common_meta_epoch = 30


pattern_r = '_ed_'+str(ed)+'_ad_'+str(ad)+'_pl_'+str(pl)+'_lr_'+str(lr)+'_pw_'+str(pw)+'_'
pdir = wdir+sample+'/'+pattern_r+'/'
pdir_r1 = wdir+sample+'/'+pattern_r+'/'+sample
pdir_r2 = wdir+sample+'/'+pattern_r+'/'+sample+'/results'

if not os.path.exists(pdir_r2):os.makedirs(pdir_r2)


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
	if batch_count >=12:
		break

picasa_object = picasa.create_picasa_object(
	batch_map,
    sample,
	'seq',
	pdir
 	)
  
params = {'device' : 'cuda',
		'batch_size' : 100,
		'input_dim' : 2000,
		'embedding_dim' : int(ed),
		'attention_dim' : int(ad),
		'latent_dim' : int(ad),
		'encoder_layers' : [100,int(ad)],
		'projection_layers' : [int(pl),int(pl)],
		'learning_rate' : float(lr),
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': float(pw),
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': common_epochs,
		'meta_epochs': common_meta_epoch
		}   



picasa_object.estimate_neighbour(params['pair_search_method'])
picasa_object.set_nn_params(params)
picasa_object.train_common()
picasa_object.plot_loss(tag='common')

device = 'cpu'
picasa_object.nn_params['device'] = device
eval_batch_size = 500
picasa_object.eval_common(eval_batch_size,device)


picasa_adata = picasa_object.result

import scanpy as sc
import matplotlib.pylab as plt
sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.pl.umap(picasa_adata,color=['batch','celltype'])
plt.savefig(pdir_r2+'/'+sample+pattern_r+'.png')