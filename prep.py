import anndata as ad
import picasa

rna = ad.read_h5ad('data/brca_scrna.h5ad')
spatial = ad.read_h5ad('data/brca_spatial.h5ad')


pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})

pico.update_common_features()

import numpy as np
mtx = np.random.random((200,100))

depth = 3
ndims = 200
replicates = 2

rp_arrs = projection_matrix(depth,ndims,replicates)

z = get_projection(mtx,rp_arrs,rp_weight_adjust=False,ndim=10)