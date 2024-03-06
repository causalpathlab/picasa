
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import picasa
from scipy.sparse import csr_matrix 


sc_ref_path = 'data/sim/brcasim_sc.h5ad'
sp_ref_path = 'data/sim/brcasim_sp.h5ad'

sc_size = 4
simdata = picasa.sim.generate_simdata(sc_ref_path,sp_ref_path,sc_size=sc_size)

sc_exp_shape = simdata['sc_exp'].shape
adata_sc = ad.AnnData(X=csr_matrix(simdata['sc_exp'].reshape(
(sc_exp_shape[0] * sc_exp_shape[1],sc_exp_shape[2]))))
adata_sc.var_names = simdata['genes']
adata_sc.obs_names = simdata['sp_nbrs'].reshape((sc_exp_shape[0] * sc_exp_shape[1],-1)).flatten()


adata_sp = ad.AnnData(X=simdata['sp_exp'])
adata_sp.var_names = simdata['genes']
adata_sp.obs_names = ['sp_'+str(x) for x in range(adata_sp.shape[0])]
adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(simdata['sp_pos']['x'],simdata['sp_pos']['y'])]

pico = picasa.create_picasa_object({'rna':adata_sc,'spatial':adata_sc})
picasa.pp.common_features(pico.data.adata_list) 

cell_indx = []
t = [ x for x in range(sc_size)]
for si in range(simdata['sp_nbrs'].shape[0]):
    cell_indx.append(np.array([x + (sc_size * si) for x in t]))
    
spsc_map ={x:y for x,y in zip(range(simdata['sp_nbrs'].shape[0]),cell_indx)}
pico.set_spsc_map(spsc_map)



pb = picasa.int.get_pairwise_interaction(spsc_map,pico.data.adata_list['rna'],rp_replicates=5,rp_depth=10,rp_dim=10,rp_weight_adjust=True)
print(pb.shape)

fname = 'figures/fig1/data/cellpair'

col_names = simdata['genes']
row_names = ['cp'+str(i) for i in range(pb.shape[0])]
smat = csr_matrix(pb)

picasa.pp.read_write.write_h5(fname,row_names,col_names,smat)


###### sim qc 

from anndata import AnnData
import scanpy as sc
import numpy as np



sc.pp.normalize_total(adata_sp, target_sum=1e4)
sc.pp.log1p(adata_sp)
dfn = adata_sp.to_df()
dfn.columns = adata_sp.var.index.values
dfn.index = adata_sp.uns['position']
    
import matplotlib.pylab as plt
from plotnine import * 

for marker in ["CD3D", "CD68", "EPCAM", "PECAM1"]:
    df = dfn[[marker]]

    df['x'] = [ float(x.split('x')[0]) for x in df.index.values]
    df['y'] = [ float(x.split('x')[1]) for x in df.index.values]

    df = pd.melt(df,id_vars=['x','y'])
    # dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)


    p = (ggplot(df, aes(x='x', y='y', color='value')) +\
    geom_point(size=5) +\
    scale_color_gradient(low="skyblue", high="yellow", name="Intensity") +\
    facet_wrap('~ variable'))

    p.save('test'+marker+'.png', dpi=300)




# pb = picasa.int.get_pairwise_interaction(spsc_map,pico.data.adata_list['rna'],rp_replicates=5,rp_depth=10,rp_dim=10,rp_weight_adjust=True)
# print(pb)

# fname = 'data/sim/sim'
# col_names = adata_sc.var.index.values
# row_names = ['cp'+str(i) for i in range(pb.shape[0])]
# smat = csr_matrix(pb)

# picasa.pp.read_write.write_h5(fname,row_names,col_names,smat)



##scanpy 
# adata = AnnData(dfexp.values)
# adata.var_names = dfexp.columns.values
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# adata = adata[:, adata.var.highly_variable]
# sc.tl.pca(adata, svd_solver='arpack')
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# sc.tl.leiden(adata,resolution=0.3)
# sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
# plt.savefig('scanpy.png');plt.close()

# sc.tl.umap(adata)
# sc.pl.umap(adata, color=['leiden'])
# plt.savefig('scanpy2.png');plt.close()
