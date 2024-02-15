
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import picasa

rna = ad.read_h5ad('data/brain/test_ad_sc.h5ad')
spatial = ad.read_h5ad('data/brain/test_ad_sp.h5ad')


pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})

picasa.pp.common_features(pico.data.adata_list) 

picasa.proj.rp(pico.data.adata_list)
picasa.proj.pmf(pico.data.adata_list)

nbr = picasa.nbr.generate_neighbours(source_adata=pico.data.adata_list['spatial'],target_adata=pico.data.adata_list['rna'],num_nbrs=10,use_projection='X_pmf_theta')


ad_map = ad.read_h5ad('../Tangram/ad_map.h5ad')
nbr2 = ad_map.X.T
nbr2 = ad_map.X
nbr2 = np.argsort(-nbr2, axis=1)[:, :10]

mtotal = 0
for ri,row in enumerate(nbr.values()):
    mtotal += len(np.intersect1d(np.array(row), np.array(nbr2[ri])))
print(mtotal)


# df_nbr = pd.DataFrame(nbr).T
# df_nbr.columns = ['spatial_index']
# df_nbr['rna_id'] = rna.obs.index.values
# df_nbr['rna_ct'] = rna.obs['celltype'].values
# spatial_classification = spatial.obs['celltype'].values
# df_nbr['spatial_ct'] = spatial_classification[df_nbr['spatial_index'].values]
# df_nbr.to_csv('data/res.csv')
# df = pd.read_csv('data/brca_celltype.csv.gz')
# rna.obs['celltype'] = pd.merge(rna.obs,df,left_index=True,right_on='cell',how='left')['celltype'].values

# dfsp = pd.read_csv('data/CID4290_metadata.csv')
# dfsp.rename(columns={'Unnamed: 0':'spot'},inplace=True)
# spatial.obs['celltype'] = pd.merge(spatial.obs,dfsp,left_index=True,right_on='spot',how='left')['Classification'].values





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

