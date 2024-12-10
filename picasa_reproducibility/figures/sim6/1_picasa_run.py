import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

import picasa
import anndata as an
import glob
import os


sample = 'sim6'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'
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
	wdir
 	)


  
params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : 2000,
		'embedding_dim' : 3000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [25,25],
		'learning_rate' : 0.001,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 12,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': 1,
		'titration': 15
		}   
  


picasa_object.estimate_neighbour(params['pair_search_method'])


def check_nbr_dist():
	import matplotlib.pylab as plt
	import seaborn as sns
	import numpy as np
	import itertools 

	dists2 = []
	for nb_pair in picasa_object.nbr_map:
		dists2.append([ x[1] for x in picasa_object.nbr_map[nb_pair].values()])

	dists2 = np.array(list(itertools.chain.from_iterable(dists2))) 
	
	sns.displot(dists)
	sns.displot(dists2)

	sns.histplot(dists, color="blue", kde=True,stat='density', label="sim5")

	sns.histplot(dists2, color="orange", kde=True,stat='density', label="sim6")

	plt.savefig(wdir+sample+'/results/dist_combine.png')
	plt.close()
	
picasa_object.set_nn_params(params)


picasa_object.train_common()
picasa_object.plot_loss(tag='common')

device = 'cpu'
picasa_object.nn_params['device'] = device
eval_batch_size = 100
picasa_object.eval_common(eval_batch_size,device)
# picasa_object.save_model()


input_dim = picasa_object.data.adata_list['Batch1'].X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = picasa_object.result.obsm['common'].shape[1]
dec_layers = [128,128]

picasa_object.train_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='unq')
eval_batch_size = 10
picasa_object.eval_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size,device='cuda')

latent_dim=15
picasa_object.train_base(input_dim, enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
picasa_object.plot_loss(tag='base')
eval_batch_size = 10
picasa_object.eval_base(input_dim, enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')

picasa_object.save_model()

