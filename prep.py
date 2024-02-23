
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import picasa


rna = ad.read_h5ad('data/brca/brca4290_scrna.h5ad')
spatial = ad.read_h5ad('data/brca/brca4290_spatial.h5ad')


pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})

picasa.pp.common_features(pico.data.adata_list) 

# picasa.proj.rp(pico.data.adata_list)
# picasa.proj.pmf(pico.data.adata_list)

# nbr = picasa.nbr.generate_neighbours(source_adata=pico.data.adata_list['spatial'],target_adata=pico.data.adata_list['rna'],num_nbrs=10,use_projection='X_pmf_theta')


# ad_map = ad.read_h5ad('../Tangram/ad_map.h5ad')
# nbr = ad_map.X.T
# nbr = np.argsort(-nbr, axis=1)[:, :10] 
# np.savez_compressed('data/spsc_map.npz', array=nbr)

nbr_data = np.load('data/spsc_map.npz')
nbr= nbr_data['array']
spsc_map ={x:y for x,y in zip(range(nbr.shape[0]),nbr)}

pico.set_spsc_map(spsc_map)

pb = picasa.int.get_pairwise_interaction(spsc_map,pico.data.adata_list['rna'],rp_replicates=5,rp_depth=10,rp_dim=10,rp_weight_adjust=True)
print(pb)

fname = 'data/cellpair'
col_names = rna.var.index.values
row_names = ['cp'+str(i) for i in range(pb.shape[0])]
smat = csr_matrix(pb)

write_h5(fname,row_names,col_names,smat)



# mtotal = 0
# for ri,row in enumerate(nbr.values()):
#     mtotal += len(np.intersect1d(np.array(row), np.array(nbr2[ri])))
# print(mtotal)


# df_nbr = pd.DataFrame(nbr).T
# df_nbr.columns = ['spatial_index']
# df_nbr['rna_id'] = rna.obs.index.values
# df_nbr['rna_ct'] = rna.obs['celltype'].values
# spatial_classification = spatial.obs['celltype'].values
# df_nbr['spatial_ct'] = spatial_classification[df_nbr['spatial_index'].values]
# df_nbr.to_csv('data/res.csv')
# df = pd.read_csv('data/brca_celltype.csv.gz')
# rna.obs['celltype'] = pd.merge(rna.obs,df,left_index=True,right_on='cell',how='left')['celltype'].values

dfsp = pd.read_csv('data/brca/CID4290_metadata.csv')
dfsp.rename(columns={'Unnamed: 0':'spot'},inplace=True)
spatial.obs['celltype'] = pd.merge(spatial.obs,dfsp,left_index=True,right_on='spot',how='left')['Classification'].values





# sc.pp.neighbors(rna,n_neighbors=10,use_rep='X_rp')
# sc.tl.umap(rna)
# sc.tl.leiden(rna)
# sc.pl.umap(rna,color=['celltype'])
# plt.savefig('data/test.png');plt.close()

# sc.pp.neighbors(spatial,n_neighbors=10,use_rep='X_rp')
# sc.tl.umap(spatial)
# sc.tl.leiden(spatial)
# sc.pl.umap(rna,color=['leiden'])
# plt.savefig('data/test2.png');plt.close()

