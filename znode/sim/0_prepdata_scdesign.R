library(scDesign3)
library(SingleCellExperiment)
library(DuoClustering2018)

sce <- get("sce_filteredExpr10_Zhengmix4eq")(metadata = FALSE)
colData(sce)$cell_type = as.factor(colData(sce)$phenoid)
colData(sce)$library = colSums(counts(sce))


set.seed(123)
example_simu <- scdesign3(
    sce = sce,
    assay_use = "counts",
    celltype = "cell_type",
    pseudotime = NULL,
    spatial = NULL,
    other_covariates = "library",
    mu_formula = "cell_type + offset(log(library))",
    sigma_formula = "1",
    family_use = "nb",
    n_cores = 2,
    usebam = FALSE,
    corr_formula = "1",
    copula = "gaussian",
    DT = TRUE,
    pseudo_obs = FALSE,
    return_model = FALSE,
    nonzerovar = FALSE,
    parallelization = "pbmcmapply",
    important_feature = "auto"
  )

  






############### sc plus spatial 

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
write.csv(df_sp, file=gzfile("sim_sp_count.csv.gz"))                              




############################



##convert to anndata

import anndata as an
import pandas as pd
import numpy as np
# import sailr

from scipy.sparse import csr_matrix 
import scanpy as sc
import matplotlib.pylab as plt

wdir = 'znode/scdesign/'

dfsc = pd.read_csv(wdir+'data/sim_sc_count.csv.gz',header=0)
dfsc = dfsc.T
dfsc.columns = dfsc.iloc[0,:]
dfsc = dfsc.iloc[1:,:]
dfsc = dfsc.astype(int)

dfscl = pd.read_csv(wdir+'data/sim_sc_label.csv.gz',header=0)


dfsp = pd.read_csv(wdir+'data/sim_sp_count.csv.gz',header=0)
dfsp = dfsp.T
dfsp.columns = dfsp.iloc[0,:]
dfsp = dfsp.iloc[1:,:]
dfsp = dfsp.astype(int)


smat_sc = csr_matrix(dfsc.to_numpy())
adata_sc = an.AnnData(X=smat_sc)
adata_sc.var_names = dfsc.columns.values
adata_sc.obs_names = dfsc.index.values
adata_sc.obs['celltype'] = dfscl.cell_type.values

smat_sp = csr_matrix(dfsp.to_numpy())
adata_sp = an.AnnData(X=smat_sp)
adata_sp.var_names = dfsp.columns.values
adata_sp.obs_names = ['sp_'+str(x) for x in dfsp.index.values]

# adata_sc.obs['celltype'] = adata_org.obs.leiden.values


adata_sp.uns['position'] = [ x.replace('sp_','') for x in adata_sp.obs_names.values]


adata_sp.write(wdir+'data/scdesign_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'data/scdesign_sc.h5ad',compression='gzip')



########## use splatter  for simulation 

# BiocManager::install("splatter")


library(splatter)
library(scater)

set.seed(1)
sce <- mockSCE()


params <- newSplatParams()

params <- setParam(params, "nGenes", 500)
params <- setParam(params, "batchCells", 3000)
params <- setParam(params, "group.prob", c(1/2,1/2))

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
adata_sp.obs_names = dfv3.index.values

# adata_sc.obs['celltype'] = adata_org.obs.leiden.values


ref_sp = an.read_h5ad(wdir+'data/sim_old_sp.h5ad')

dfspl = ref_sp.uns['sp_pos'].sample(adata_sp.X.shape[0]).reset_index(drop=True)
dfspl = dfspl[['x','y']]

adata_sp.uns['position'] = [ str(x)+'x'+str(y) for x,y in zip(dfspl['x'],dfspl['y'])]


adata_sp.write(wdir+'data/sim_sp.h5ad',compression='gzip')
adata_sc.write(wdir+'data/sim_sc.h5ad',compression='gzip')
