###### sim qc 

from anndata import an
import scanpy as sc
import numpy as np
import matplotlib.pylab as plt
from plotnine import * 

spatial = an.read_h5ad('data/sim/sim_sp.h5ad')

sc.pp.normalize_total(adata_sp, target_sum=1e4)
sc.pp.log1p(adata_sp)
dfn = adata_sp.to_df()
dfn.columns = adata_sp.var.index.values
dfn.index = adata_sp.uns['position']
    

for marker in ["CD3D", "CD68", "EPCAM", "PECAM1"]:
    df = dfn[[marker]]

    df['x'] = [ float(x.split('x')[0]) for x in df.index.values]
    df['y'] = [ float(x.split('x')[1]) for x in df.index.values]

    df = pd.melt(df,id_vars=['x','y'])
    # dfn['value'] = dfn['value'].apply( lambda x: 1 if x>1 else x)


    p = (ggplot(df, aes(x='x', y='y', color='value')) +\
    geom_point(size=1) +\
    scale_color_gradient(low="skyblue", high="yellow", name="Intensity") +\
    facet_wrap('~ variable'))

    p.save('test'+marker+'.png', dpi=300)



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
