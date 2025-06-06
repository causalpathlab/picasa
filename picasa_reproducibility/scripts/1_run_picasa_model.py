
import picasa
import anndata as an
import glob
import os


sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'

common_epochs = 1
common_meta_epoch = 15
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
	if batch_count >=25:
		break

picasa_object = picasa.create_picasa_object(
	batch_map,
    sample,
	'seq',
	wdir
 	)

params = {'device' : 'cuda',
		'batch_size' : 100,
		'input_dim' : 2037,
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
  





picasa_object.estimate_neighbour(params['pair_search_method'])

picasa_object.set_nn_params(params)


picasa_object.train_common()

# picasa_object.plot_loss(tag='common')
device = 'cpu'
picasa_object.nn_params['device'] = device
eval_batch_size = 500
picasa_object.eval_common(eval_batch_size,device)

### get full data for unique and base model training

adata = an.read_h5ad(wdir+sample+'/data/all_brca.h5ad')

enc_layers = [128,25]
unique_latent_dim = params['latent_dim']
common_latent_dim = params['latent_dim']
dec_layers = [128,128]

picasa_object.train_unique(adata,enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=unique_epoch,batch_size=128,device='cuda')
# picasa_object.plot_loss(tag='unq')


eval_batch_size = 1000
picasa_object.eval_unique(adata,enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size,device='cuda')

latent_dim=params['latent_dim']
picasa_object.train_base(adata,enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=base_epoch,batch_size=128,device='cuda')
# picasa_object.plot_loss(tag='base')

eval_batch_size = 500
picasa_object.eval_base(adata,enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')
picasa_object.save_model()

