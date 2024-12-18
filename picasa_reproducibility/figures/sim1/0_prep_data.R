#!/usr/bin/env Rscript

n = 1
require(splatter)
#' simulation_1 scenario 1 
dir.create("simulation_1")

# generate seed 
set.seed(123)
sim1.seeds = sample(seq.int(100, 9999), n)

for (i in seq.int(n) ){
    s = sim1.seeds[i]

    params <- newSplatParams()
    params <- setParam(params, "nGenes", 1000)
    params <- setParam(params, "batchCells", c(3000,2900,2800))
    params <- setParam(params, "group.prob", c(1/5,1/5,1/5,1/5,1/5))


    params <- setParam(params, "batch.facLoc", 0.1)  # Increase the location effect
    params <- setParam(params, "batch.facScale", 0.1)  # Increase the scale effect


    sim <- splatSimulate(params, method="groups", verbose=FALSE)
                      
    saveRDS(sim, paste0( "simulation_1/simData_", i, ".RDS") )
}
 



 
 
# Load necessary libraries
library(SingleCellExperiment)

n=1
for (i in seq.int(n) ){

# Load the RDS file
scdata <- readRDS(paste0('simulation_nested/simData_',i,'.RDS'))

# Extract the counts data
counts_data <- assay(scdata, "counts")

# Extract row and column metadata
row_data <- as.data.frame(rowData(scdata))
col_data <- as.data.frame(colData(scdata))

# Save counts data to CSV
write.csv(as.data.frame(counts_data), paste0('simulation_nested/counts_data_',i,'.csv'), row.names = FALSE)

# Save row and column metadata to CSV
write.csv(row_data, paste0('simulation_nested/row_data_',i,'.csv'), row.names = FALSE)
write.csv(col_data, paste0('simulation_nested/col_data_',i,'.csv'), row.names = FALSE)

}
