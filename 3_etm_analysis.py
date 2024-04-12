
import matplotlib.pylab as plt
import anndata as an
import pandas as pd
import numpy as np
import sailr
import torch
import umap


### predict for test data

device = 'cpu'
input_dims = 20309
latent_dims = 10
encoder_layers = [200,100,10]
l_rate = 0.01
epochs= 500

wdir = 'node/sim/'
rna = an.read_h5ad(wdir+'data/sim_sc.h5ad')
        
batch_size=4113
data_pred = sailr.du.nn_load_data(rna,device,batch_size)
sailr_model = sailr.nn_etm.SAILRNET(input_dims, latent_dims, encoder_layers).to(device)
sailr_model.load_state_dict(torch.load(wdir+'results/nn_etm.model'))
m,ylabel = sailr.nn_etm.predict(sailr_model,data_pred)



### plot theta and beta
from sailr.util.plots import plot_umap_df,plot_gene_loading

dfh = pd.DataFrame(m.theta.cpu().detach().numpy())
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.5,metric='cosine')
proj_2d = umap_2d.fit(dfh)
df_umap= pd.DataFrame()
df_umap['cell'] = ylabel
df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


# df_umap['celltype'] = [x.split('_')[2] for x in df_umap['cell']]

dfl = pd.read_csv(wdir+'data/sim_meta.tsv',sep='\t')
dfl = dfl[['Cell','Celltype (major-lineage)']]
dfl.columns = ['cell','celltype']
df_umap['celltype'] = pd.merge(df_umap,dfl, on='cell')['celltype'].values

plot_umap_df(df_umap,'celltype',wdir+'results/nn_etm',pt_size=1.0,ftype='png')

dfbeta = pd.DataFrame(m.beta.cpu().detach().numpy())
dfbeta.columns = rna.var.index.values
plot_gene_loading(dfbeta,top_n=5,max_thresh=100,fname=wdir+'results/beta')



# mats = [
#         m.z_sc.cpu().detach().numpy(),
#         m.theta.cpu().detach().numpy(),
#         m.beta.cpu().detach().numpy(),
#         ]
# mats_name = ['z_sc','theta','beta']


# fig, axes = plt.subplots(1, 3, figsize=(25, 15))

# for i in range(1):
#     for j in range(3):
#         idx =  j
#         ax = axes[j]  
#         print(idx)
#         heatmap = ax.imshow(mats[idx], cmap='viridis', aspect='auto')
#         ax.set_title(f'{mats_name[idx]}')
#         fig.colorbar(heatmap, ax=ax) 

# plt.tight_layout()  
# plt.savefig('testnn_etm.png');plt.close()

