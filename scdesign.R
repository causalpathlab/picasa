library(scDesign3)
library(SingleCellExperiment)



example_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581980")))
print(example_sce)



##### spatial + scrna 

MOBSC_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581983")))
MOBSP_sce <- readRDS((url("https://figshare.com/ndownloader/files/40581986")))
print(MOBSC_sce)
print(MOBSP_sce)

cell_type <- unique(colData(MOBSC_sce)$cellType)



set.seed(123)
MOBSC_data <- construct_data(
  sce = MOBSC_sce,
  assay_use = "counts",
  celltype = "cell_type",
  pseudotime = NULL,
  spatial = NULL,
  other_covariates = NULL,
  corr_by = "1"
)

MOBSC_marginal <- fit_marginal(
  data = MOBSC_data,
  predictor = "gene",
  mu_formula = "cell_type",
  sigma_formula = "cell_type",
  family_use = "nb",
  n_cores = 2,
  usebam = FALSE,
  parallelization = "pbmcmapply"

)

MOBSC_copula <- fit_copula(
  sce = MOBSC_sce,
  assay_use = "counts",
  marginal_list = MOBSC_marginal,
  family_use = "nb",
  copula = "gaussian",
  n_cores = 2,
  input_data = MOBSC_data$dat
)

MOBSC_para <- extract_para(
  sce = MOBSC_sce,
  marginal_list = MOBSC_marginal,
  n_cores = 2,
  family_use = "nb",
  new_covariate = MOBSC_data$newCovariate,
  data = MOBSC_data$dat
)

MOBSC_newcount <- simu_new(
  sce = MOBSC_sce,
  mean_mat = MOBSC_para$mean_mat,
  sigma_mat = MOBSC_para$sigma_mat,
  zero_mat = MOBSC_para$zero_mat,
  quantile_mat = NULL,
  copula_list = MOBSC_copula$copula_list,
  n_cores = 2,
  family_use = "nb",
  input_data = MOBSC_data$dat,
  new_covariate = MOBSC_data$newCovariate,
  filtered_gene = MOBSC_data$filtered_gene
)



df_sc = as.data.frame(as.matrix(MOBSC_newcount))
write.csv(df_sc, file=gzfile("sim_sc_count.csv.gz"))                              
write.csv(MOBSC_data$newCovariate, file=gzfile("sim_sc_label.csv.gz"))                 




########## spatial

set.seed(123)
MOBSP_data <- construct_data(
  sce = MOBSP_sce,
  assay_use = "counts",
  celltype = NULL,
  pseudotime = NULL,
  spatial = c("spatial1", "spatial2"),
  other_covariates = NULL,
  corr_by = "1"
)

MOBSP_marginal <- fit_marginal(
  data = MOBSP_data,
  predictor = "gene",
  mu_formula = "s(spatial1, spatial2, bs = 'gp', k = 50, m = c(1, 2, 1))",
  sigma_formula = "1",
  family_use = "nb",
  n_cores = 2,
  usebam = FALSE, 
  parallelization = "pbmcmapply"
  
)

MOBSP_copula <- fit_copula(
  sce = MOBSP_sce,
  assay_use = "counts",
  marginal_list = MOBSP_marginal,
  family_use = "nb",
  copula = "gaussian",
  n_cores = 2,
  input_data = MOBSP_data$dat
)

MOBSP_para <- extract_para(
  sce = MOBSP_sce,
  marginal_list = MOBSP_marginal,
  n_cores = 2,
  family_use = "nb",
  new_covariate = MOBSP_data$newCovariate,
  data = MOBSP_data$dat
)


MOBSC_sig_matrix <- sapply(cell_type, function(x) {
  rowMeans(t(MOBSC_para$mean_mat)[, colData(MOBSC_sce)$cellType %in% x])
})

MOBSP_matrix <- (t(MOBSP_para$mean_mat))


sig_matrix <- as.data.frame(MOBSC_sig_matrix)
mixture_file <- as.data.frame(MOBSP_matrix)



#####


set.seed(123)
n_rows <- 278
n_cols <- 4

proportion_mat <- data.frame(matrix(nrow = n_rows, ncol = n_cols))

for (i in 1:n_rows) {
  props <- runif(n_cols, min = 0, max = 0.1)  
  index <- sample(n_cols, 1)
  props[index] <- runif(1, min = 0.9, max = 1)
  proportion_mat[i, ] <- props / sum(props)
}
colnames(proportion_mat) = cell_type

# #####
# proportion_mat = t(proportion_mat)

set.seed(123)
MOBSCSIM_sce <- MOBSC_sce
counts(MOBSCSIM_sce) <- MOBSC_newcount


## 50 cell per spot and then divide by 5 to get 10 cells per spot 
MOBSP_new_mixture <- (apply(proportion_mat, 1, function(x) {
  n = 50
  rowSums(sapply(cell_type, function(y) {
    index <- sample(which(colData(MOBSCSIM_sce)$cell_type==y), size = n, replace = FALSE)
    rowSums(MOBSC_newcount[, index])*x[y]
  }))
}))


MOBSP_new_mixture <- MOBSP_new_mixture/5

### Ceiling to integer
MOBSP_new_mixture <- ceiling(MOBSP_new_mixture)


df_sp = as.data.frame(as.matrix(MOBSP_new_mixture))
colnames(df_sp) = colnames(MOBSP_sce)
write.csv(df_sc, file=gzfile("sim_sp_count.csv.gz"))                              




############################



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

params <- setParam(params, "nGenes", 500)
params <- setParam(params, "batchCells", c(2000,2100,2200))
params <- setParam(params, "group.prob", c(1/3,1/3,1/3))

sim <- splatSimulate(params, method="groups", verbose=FALSE)

counts = data.frame(counts(sim))
meta = data.frame(colData(sim))

write.csv(counts, file=gzfile("sim_count.csv.gz"))
write.csv(meta, file=gzfile("sim_label.csv.gz"))                 

##convert to anndata

import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt


df = pd.read_csv('sim_count.csv.gz',header=0)
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]
df = df.astype(int)


dfl = pd.read_csv('sim_label.csv.gz',header=0)

smat = csr_matrix(df.to_numpy())
adata = an.AnnData(X=smat)
adata.var_names = df.columns.values
adata.obs_names = df.index.values
adata.obs['celltype'] = dfl['Group'].values
adata.obs['Batch'] = dfl['Batch'].values
adata.write('sim2.h5ad',compression='gzip')
