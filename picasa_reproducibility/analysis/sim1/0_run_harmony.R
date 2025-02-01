
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("harmony")

library(SingleCellExperiment)
library(harmony)
library(scater)

# Load simulated data
scdata <- readRDS("simulation/simData_1.RDS")

# Normalize and perform PCA
scdata <- logNormCounts(scdata)
scdata <- runPCA(scdata, ncomponents = 50, exprs_values = "logcounts")

# Batch integration with Harmony
batch_key <- "Batch"  # Update with the correct column name for batches
harmony_result <- HarmonyMatrix(
  reducedDim(scdata, "PCA"),
  meta_data = as.data.frame(colData(scdata)),
  do_pca = FALSE, 
  vars_use = batch_key
)

# Update PCA embeddings with Harmony results
reducedDim(scdata, "PCA") <- harmony_result

# UMAP visualization
scdata <- runUMAP(scdata, dimred = "PCA")
plotUMAP(scdata, colour_by = batch_key)

# Save the integrated PCA matrix and metadata
write.csv(as.data.frame(reducedDim(scdata, "PCA")), "simulation/harmony_integrated_PCA.csv")
write.csv(as.data.frame(colData(scdata)), "simulation/harmony_integrated_metadata.csv")
