import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import picasa
import anndata as an



sample = 'sim4'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/'
ddir = wdir+sample+'/data/'
batch1 = an.read_h5ad(ddir+sample+'_Batch1.h5ad')
batch2 = an.read_h5ad(ddir+sample+'_Batch2.h5ad')
batch3 = an.read_h5ad(ddir+sample+'_Batch3.h5ad')

picasa_object = picasa.create_picasa_object(
	{'Batch1':batch1,
	 'Batch2':batch2,
	 'Batch3':batch3
	 },
    'sim4',
	'seq',
	wdir
 	)


params = {'device' : 'cuda',
		'batch_size' : 64,
		'input_dim' : 1000,
		'embedding_dim' : 1000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [15,15],
		'learning_rate' : 0.001,
		'lambda_loss' : [1.0,0.1,0.0,1.0],
		'temperature_cl' : 1.0,
		'pair_search_method' : 'approx_50',
        'pair_importance_weight': 0.01,
	 	'corruption_tol' : 10.0,
        'cl_loss_mode' : 'none', 
      	'loss_clusters' : 5, 
        'loss_threshold' : 0.1, 
        'loss_weight': 2.0,
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
eval_total_size = 50000
picasa_object.eval_common(eval_batch_size,eval_total_size,device)

picasa_object.set_batch_mapping()


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

