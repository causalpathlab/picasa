#!/usr/bin/env Rscript

require(splatter)
require(scater)
require(jsonlite)

# Directory for saving data
dir.create("simulation", showWarnings = FALSE)

# Simulation setup
nGenes <- 1000
nCellsPerBatch <- 1250
cellTypeProportions <- c(0.2, 0.2, 0.2, 0.2, 0.2)
batchEffects <- list(
    list(name = "Batch1", facLoc = 0.2, facScale = 0.1),
    list(name = "Batch2", facLoc = 0.5, facScale = 0.3)
)
conditionEffects <- list(
    list(name = "Control", deProb = 0.2, deFacLoc = 1.5),
    list(name = "Treated", deProb = 0.4, deFacLoc = 2.0)
)

# Initialize storage
all_batches <- list()

# Simulate data for each batch and condition
for (batch in batchEffects) {
    for (condition in conditionEffects) {
        params <- newSplatParams()
        params <- setParam(params, "nGenes", nGenes)
        params <- setParam(params, "batchCells", nCellsPerBatch)
        params <- setParam(params, "group.prob", cellTypeProportions)
        params <- setParam(params, "batch.facLoc", batch$facLoc)
        params <- setParam(params, "batch.facScale", batch$facScale)
        params <- setParam(params, "de.prob", condition$deProb)
        params <- setParam(params, "de.facLoc", condition$deFacLoc)

        sim <- splatSimulate(params, method = "groups", verbose = FALSE)
        colData(sim)$Batch <- batch$name
        colData(sim)$Condition <- condition$name
        all_batches <- append(all_batches, list(sim))
    }
}

# Standardize row metadata before merging
for (i in seq_along(all_batches)) {
    if (i == 1) {
        reference_row_data <- rowData(all_batches[[i]])
    } else {
        rowData(all_batches[[i]]) <- reference_row_data
    }
}

# Merge all simulations
mergedSim <- do.call(cbind, all_batches)

# Save the simulation
saveRDS(mergedSim, "simulation/mergedSim.RDS")


# Load necessary libraries
library(SingleCellExperiment)

n=1
for (i in seq.int(n) ){

# Load the RDS file
scdata <- readRDS(paste0('simulation/simData_',i,'.RDS'))

# Extract the counts data
counts_data <- assay(scdata, "counts")

# Extract row and column metadata
row_data <- as.data.frame(rowData(scdata))
col_data <- as.data.frame(colData(scdata))

# Save counts data to CSV
write.csv(as.data.frame(counts_data), paste0('simulation/counts_data_',i,'.csv'), row.names = FALSE)

# Save row and column metadata to CSV
write.csv(row_data, paste0('simulation/row_data_',i,'.csv'), row.names = FALSE)
write.csv(col_data, paste0('simulation/col_data_',i,'.csv'), row.names = FALSE)

}
