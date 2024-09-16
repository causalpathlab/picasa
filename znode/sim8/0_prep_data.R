library(scDesign3)
library(SingleCellExperiment)


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


df_sc = as.data.frame(as.matrix(simu_res$new_count))

write.csv(df_sc, file=gzfile("data/sim_sc_count.csv.gz"))                              
write.csv(simu_res$new_covariate, file=gzfile("data/sim8_label.csv.gz"))  

