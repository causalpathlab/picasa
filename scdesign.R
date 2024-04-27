example_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581965")))
print(example_sce)

example_sce <- example_sce[1:1000, ]

set.seed(123)
simu_res <- scdesign3(sce = example_sce, 
                              assay_use = "counts", 
                              celltype = "cell_type", 
                              pseudotime = NULL, 
                              spatial = NULL, 
                              other_covariates = c("batch"), 
                              mu_formula = "cell_type + batch", 
                              sigma_formula = "1", 
                              family_use = "nb", 
                              n_cores = 2, 
                              usebam = FALSE, 
                              corr_formula = "1", 
                              copula = "gaussian", 
                              DT = TRUE, 
                              pseudo_obs = FALSE, 
                              return_model = FALSE)

write.csv(simu_res$new_count, file=gzfile("pbmc_count.csv.gz"))                              
write.csv(simu_res$new_covariate, file=gzfile("pbmc_label.csv.gz"))                 

##convert to anndata

import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt

wdir = 'znode/pbmc/'

df = pd.read_csv(wdir+'data/pbmc_count.csv.gz',header=0)
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]
df = df.astype(int)

dfv2 = df.iloc[df.index.str.contains('V2'),:]
dfv3 = df.iloc[df.index.str.contains('v3'),:]

smatv2 = csr_matrix(dfv2.to_numpy())
adata_sc = an.AnnData(X=smatv2)
adata_sc.var_names = dfv2.columns.values
adata_sc.obs_names = dfv2.index.values

smatv3 = csr_matrix(dfv3.to_numpy())
adata_sp = an.AnnData(X=smatv3)
adata_sp.var_names = dfv3.columns.values
adata_sp.obs_names = ['sp_'+str(x) for x in dfv3.index.values]

# adata_sc.obs['celltype'] = adata_org.obs.leiden.values


ref_sp = an.read_h5ad('node/sim/data/sim_sp.h5ad')

dfspl = ref_sp.uns['sp_pos'].sample(adata_sp.X.shape[0]).reset_index(drop=True)
dfspl = dfspl[['x','y']]

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(dfspl['x'],dfspl['y'])]

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

distmat =  cdist(adata_sc.X.todense(), adata_sp.X.todense())
sorted_indices = np.argsort(distmat, axis=1)
distdf = pd.DataFrame(sorted_indices)

f = [x for x in range(0,25)]
l = [x for x in range(distdf.shape[1]-25,distdf.shape[1])]
distdf = distdf[f+l]

distdf.to_csv(wdir+'data/sc_sp_dist.csv.gz',index=False,compression='gzip')


adata_sp.write(wdir+'data/pbmc_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'data/pbmc_sc.h5ad',compression='gzip')


smat = csr_matrix(df.to_numpy())
adata_sc = an.AnnData(X=smat)
adata_sc.var_names = df.columns.values
adata_sc.obs_names = df.index.values

adata_sc.write(wdir+'data/pbmc_sc_2b.h5ad',compression='gzip')



########## use splatter  for simulation 

# BiocManager::install("splatter")


library(splatter)

params <- newSplatParams()

params <- setParam(params, "nGenes", 1000)
params <- setParam(params, "batchCells", c(3000,2900))
params <- setParam(params, "group.prob", c(1/2,1/2))

sim <- splatSimulate(params, method="groups", verbose=FALSE)

counts = data.frame(counts(sim))
meta = data.frame(colData(sim))

write.csv(counts, file=gzfile("znode/sim/data/sim_count.csv.gz"))
write.csv(meta, file=gzfile("znode/sim/data/sim_label.csv.gz"))                 

##convert to anndata

import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


wdir = 'znode/sim/'

df = pd.read_csv(wdir+'data/sim_count.csv.gz',header=0)
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]
df = df.astype(int)


dfl = pd.read_csv(wdir+'data/sim_label.csv.gz',header=0)

df.index = [x+'_'+y for x,y in zip(dfl.Batch.values,df.index.values)]
df.index = [x+'_'+y for x,y in zip(dfl.Group.values,df.index.values)]

dfv2 = df.iloc[(dfl['Batch']=='Batch1').values,:]
dfv3 = df.iloc[(dfl['Batch']=='Batch2').values,:]

smatv2 = csr_matrix(dfv2.to_numpy())
adata_sc = an.AnnData(X=smatv2)
adata_sc.var_names = dfv2.columns.values
adata_sc.obs_names = dfv2.index.values

smatv3 = csr_matrix(dfv3.to_numpy())
adata_sp = an.AnnData(X=smatv3)
adata_sp.var_names = dfv3.columns.values
adata_sp.obs_names = ['sp_'+str(x) for x in dfv3.index.values]

# adata_sc.obs['celltype'] = adata_org.obs.leiden.values


ref_sp = an.read_h5ad('znode/sim_old/data/sim_sp.h5ad')

dfspl = ref_sp.uns['sp_pos'].sample(adata_sp.X.shape[0]).reset_index(drop=True)
dfspl = dfspl[['x','y']]

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(dfspl['x'],dfspl['y'])]

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

distmat =  cdist(adata_sc.X.todense(), adata_sp.X.todense())
sorted_indices = np.argsort(distmat, axis=1)
distdf = pd.DataFrame(sorted_indices)

f = [x for x in range(0,25)]
l = [x for x in range(distdf.shape[1]-25,distdf.shape[1])]
distdf = distdf[f+l]

distdf.to_csv(wdir+'data/sc_sp_dist.csv.gz',index=False,compression='gzip')


adata_sp.write(wdir+'data/sim_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'data/sim_sc.h5ad',compression='gzip')
