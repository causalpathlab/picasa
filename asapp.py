import asappy
import anndata as an

######################################################
sample = 'cellpair'

wdir = 'figures/fig1/'

data_size = 110000
number_batches = 1


asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)



asappy.generate_pseudobulk(asap_object,tree_depth=10)

n_topics = 25 ## paper 
asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)

# asap_adata = asappy.generate_model(asap_object,return_object=True)
asappy.generate_model(asap_object)

asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

## top 10 main paper
asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
	
cluster_resolution= 0.1 ## paper
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
print(asap_adata.obs.cluster.value_counts())
	
## min distance 0.5 paper
asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
asappy.plot_umap(asap_adata,col='cluster',pt_size=0.5,ftype='png')

asap_adata.write(wdir+'results/'+sample+'.h5asapad')
	
