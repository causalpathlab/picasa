
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import picasa
from scipy.sparse import csr_matrix 

sc_ref_path = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/fig1/data/sc.h5ad'
sp_ref_path = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/fig1/data/sp.h5ad'

dfsc,dfsp,nbrs = picasa.sim.generate_simdata(sc_ref_path,sp_ref_path)
# dfsp.to_csv('dataspmap.csv')

adata_sc = ad.AnnData(X=csr_matrix(dfsc.values))
adata_sc.var_names = dfsc.columns
adata_sc.obs_names = dfsc.index.values

adata_sp = ad.read_h5ad(sp_ref_path)

pico = picasa.create_picasa_object({'rna':adata_sc,'spatial':adata_sc})
picasa.pp.common_features(pico.data.adata_list) 


nbrs = np.array(nbrs)
spsc_map ={x:y for x,y in zip(range(nbrs.shape[0]),nbrs)}
pico.set_spsc_map(spsc_map)


from picasa.dutil.data import load_data
def get_nbr_mtx(sc_adata: AnnData, spsc_map: dict) -> np.array:
    mtx = []
    for nbrs in spsc_map.values():
        cmtx = []
        for nbr_index in nbrs:
            nbr_exp = load_data(adata_sc,nbr_index,nbr_index+1)
            cmtx.append(nbr_exp)
        cmtx = np.squeeze(np.array(cmtx))
        mtx.append(cmtx.mean(0))
    return np.squeeze(np.array(mtx))

dfexp = pd.DataFrame(get_nbr_mtx(adata_sc,spsc_map))
dfexp.index = adata_sp.obs.index.values
dfexp.columns = adata_sp.var.index.values


from anndata import AnnData
import scanpy as sc
import numpy as np


adata = AnnData(dfexp.values)
# sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dfn = adata.to_df()
dfn.columns = dfexp.columns
dfn.index = dfexp.index.values
    
dfn = dfn[["Penk", "Apold1", "Cdhr1", "S100a5"]]

dfn['x'] = [ float(x.split('x')[0]) for x in dfn.index.values]
dfn['y'] = [ float(x.split('x')[1]) for x in dfn.index.values]
print(dfn.sum())

dfn = pd.melt(dfn,id_vars=['x','y'])
dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)

import matplotlib.pylab as plt
from plotnine import * 

ggplot(dfn, aes(x='x', y='y', color='value')) +\
geom_point(size=2) +\
scale_color_gradient(low="skyblue", high="yellow", name="Intensity") +\
facet_wrap('~ variable')

plt.savefig('test.png');plt.close()




# pb = picasa.int.get_pairwise_interaction(spsc_map,pico.data.adata_list['rna'],rp_replicates=5,rp_depth=10,rp_dim=10,rp_weight_adjust=True)
# print(pb)

# fname = 'data/sim/sim'
# col_names = adata_sc.var.index.values
# row_names = ['cp'+str(i) for i in range(pb.shape[0])]
# smat = csr_matrix(pb)

# picasa.pp.read_write.write_h5(fname,row_names,col_names,smat)

