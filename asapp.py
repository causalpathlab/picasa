import asappy
import pandas as pd
import numpy as np
import asapc
import anndata as ad


from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from plotnine import * 


######################################################
sample = 'cellpair'

wdir = 'figures/fig1/'

data_size = 110000
number_batches = 1


# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)



asappy.generate_pseudobulk(asap_object,tree_depth=10)

n_topics = 10 ## paper 
asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)

# asap_adata = asappy.generate_model(asap_object,return_object=True)
asappy.generate_model(asap_object)

asap_adata = ad.read_h5ad(wdir+'results/'+sample+'.h5asap')


## top 10 main paper
asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
	
# cluster_resolution= 0.1 ## paper
# asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
# print(asap_adata.obs.cluster.value_counts())
	
# ## min distance 0.5 paper
# asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
# asappy.plot_umap(asap_adata,col='cluster',pt_size=0.5,ftype='png')

# asap_adata.write(wdir+'results/'+sample+'.h5asapad')
	


adata_sp = ad.read_h5ad('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/data/sim/brcasim_sp.h5ad')

df = pd.DataFrame(adata_sp.X)
df.columns = adata_sp.var.index.values
f = df[asap_adata.var.index.values]

pred_model = asapc.ASAPaltNMFPredict(df.values.T,asap_adata.varm['beta_log_scaled'])
pred = pred_model.predict()
corr = pred.corr
theta = pred.theta

kmeans = KMeans(n_clusters=10, random_state=0).fit(theta)
dfn = pd.DataFrame(kmeans.labels_)
dfn.columns = ['cluster']
dfn['cluster'] = ['c'+str(i) for i in dfn['cluster']]
dfn.index = adata_sp.obs['position']
    

for marker in ["cluster"]:
    df = dfn[[marker]]

    df['x'] = [ float(x.split('x')[0]) for x in df.index.values]
    df['y'] = [ float(x.split('x')[1]) for x in df.index.values]

    df = pd.melt(df,id_vars=['x','y'])
    # dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)


    p = (ggplot(df, aes(x='x', y='y', color='value')) +\
    geom_point(size=5) +\
    facet_wrap('~ variable'))

    p.save('test'+marker+'theta.png', dpi=300)

kmeans = KMeans(n_clusters=10, random_state=0).fit(corr)
dfn = pd.DataFrame(kmeans.labels_)
dfn.columns = ['cluster']
dfn['cluster'] = ['c'+str(i) for i in dfn['cluster']]
dfn.index = adata_sp.obs['position']
    

for marker in ["cluster"]:
    df = dfn[[marker]]

    df['x'] = [ float(x.split('x')[0]) for x in df.index.values]
    df['y'] = [ float(x.split('x')[1]) for x in df.index.values]

    df = pd.melt(df,id_vars=['x','y'])
    # dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)


    p = (ggplot(df, aes(x='x', y='y', color='value')) +\
    geom_point(size=5) +\
    facet_wrap('~ variable'))

    p.save('test'+marker+'corr.png', dpi=300)



dfn = pd.DataFrame(corr)
dfn.index = adata_sp.obs['position']
    

for marker in dfn.columns:
    df = dfn[[marker]]

    df['x'] = [ float(x.split('x')[0]) for x in df.index.values]
    df['y'] = [ float(x.split('x')[1]) for x in df.index.values]

    df = pd.melt(df,id_vars=['x','y'])
    # dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)


    p = (ggplot(df, aes(x='x', y='y', color='value')) +\
    geom_point(size=5) +\
    facet_wrap('~ variable'))

    p.save('test'+str(marker)+'corr.png', dpi=300)

