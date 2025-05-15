import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

import picasa
import anndata as an
import glob
import os


sample = 'sim3'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


common_epochs = 1
common_meta_epoch = 17
unique_epoch = 250
base_epoch = 250

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
	wdir+sample
 	)


  

  
params = {'device' : 'cuda',
		'batch_size' : 100,
		'input_dim' : 2000,
		'embedding_dim' : 2750,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [25,25],
		'learning_rate' : 0.001,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 0.75,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': common_epochs,
		'meta_epochs': common_meta_epoch
		}   



picasa_object.estimate_neighbour(params['pair_search_method'])


picasa_object.set_nn_params(params)


# picasa_object.train_common()
# picasa_object.plot_loss(tag='common')
# device = 'cpu'
# picasa_object.nn_params['device'] = device
# eval_batch_size = 500
# picasa_object.eval_common(eval_batch_size,device)



picasa_adata = an.read_h5ad(wdir+sample+'/results/picasa.h5ad')
picasa_object.prep_previous_common_model(picasa_adata.obsm['common'])


adata_combined = an.read_h5ad(ddir+'all_sim3.h5ad')
input_dim = adata_combined.X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = picasa_object.result.obsm['common'].shape[1]
dec_layers = [128,128]




picasa_object.train_unique(adata_combined, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=unique_epoch,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='unq')
eval_batch_size = 10
picasa_object.eval_unique(adata_combined, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size,device='cuda')

latent_dim=picasa_object.result.obsm['common'].shape[1]
picasa_object.train_base(adata_combined, enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=base_epoch,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='base')
eval_batch_size = 10
picasa_object.eval_base(adata_combined, enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')
picasa_object.save_model()

