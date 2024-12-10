import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import picasa
import anndata as an



sample = 'pbmc'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'
ddir = wdir+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_batch_0.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_batch_1.h5ad')

picasa_object = picasa.create_picasa_object(
	{'pbmc1':batch1,
	 'pbmc2':batch2,
	 },
    'pbmc',
	'seq',
	wdir
 	)


params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : 1199,
		'embedding_dim' : 2000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 20,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
		'epochs': 1,
		'titration': 15
		}
  
picasa_object.estimate_neighbour(params['pair_search_method'])
	
picasa_object.set_nn_params(params)


picasa_object.train_common()
picasa_object.plot_loss(tag='common')

device = 'cpu'
picasa_object.nn_params['device'] = device
eval_batch_size = 300
eval_total_size = 5000
picasa_object.eval_common(eval_batch_size,eval_total_size,device)

picasa_object.set_batch_mapping()
picasa_object.save_model()


# input_dim = picasa_object.data.adata_list['pbmc1'].X.shape[1]
# enc_layers = [128,15]
# unique_latent_dim = 15
# common_latent_dim = picasa_object.result.obsm['common'].shape[1]
# dec_layers = [128,128]

# picasa_object.train_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
# picasa_object.plot_loss(tag='unq')
# eval_batch_size = 10
# picasa_object.eval_unique(input_dim, enc_layers,common_latent_dim,unique_latent_dim,dec_layers,eval_batch_size,device='cuda')

# latent_dim=15
# picasa_object.train_base(input_dim, enc_layers,latent_dim,dec_layers,l_rate=0.001,epochs=250,batch_size=128,device='cuda')
# picasa_object.plot_loss(tag='base')
# eval_batch_size = 10
# picasa_object.eval_base(input_dim, enc_layers,latent_dim,dec_layers,eval_batch_size,device='cuda')

# picasa_object.save_model()

