#!/usr/bin/env Rscript

splatPopSim <- function(vcf.n.samples = 5, vcf.n.snps = 2000, 
			bulk.n.genes = 2000, bulk.n.samples = 50,
			group.prob, nGenes = 2000, batchCells = 500, 
			group2samples = NULL, sampling.n = NULL, 
			seed = 123, ...){

    require(splatter)
    require(scater)
    require(VariantAnnotation)

    vcf <- mockVCF(n.samples = vcf.n.samples, n.snps = vcf.n.snps, seed = seed)
    bulk.eqtl <- mockBulkeQTL(n.genes = bulk.n.genes, seed = seed)
    bulk.means <- mockBulkMatrix(n.genes = bulk.n.genes, n.samples = bulk.n.samples, seed = seed)
    
    params.est <- splatPopEstimate(means = bulk.means,
                                   eqtl = bulk.eqtl,
                                   counts = mockSCE()
    				   )
    
    params.est <- setParams(params.est, nGenes = nGenes, group.prob = group.prob, batchCells = batchCells, similarity.scale = 6)
    
    sim.sc.est <- splatPopSimulate(vcf = vcf, params = params.est, seed = seed)

    if (!is.null(group2samples ) ){
	keep = sapply(names(group2samples), function(x) sim.sc.est$Group == x & sim.sc.est$Sample %in% group2samples[[x]] ) 	    	 
        
    	sim.sc.est = sim.sc.est[, rowSums(keep) == 1]    
    }

    sim.annot = colData(sim.sc.est) 

    if(!is.null(sampling.n) ){
	    require(sampling)
	    set.seed(seed)
	    sampling.n = sampling.n[unique(sim.annot$Sample) ]

 	    sim.annot.ds = sampling::strata(sim.annot, stratanames = "Sample", size = sampling.n, method = "srswor")	
        sim.sc.est = sim.sc.est[, sim.annot.ds$ID_unit]    
    }

    return(sim.sc.est)
}

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
                      
    # sim = splatPopSim(vcf.n.samples = 6,
    #                   vcf.n.snps = 20000,
    #                   bulk.n.genes = 2000,
    #                   bulk.n.samples = 100,
    #                   group.prob = rep(1/5, 5),
    #                   nGenes = 2000,
    #                   batchCells = 500,
    #                   seed = s
    #                   )

    saveRDS(sim, paste0( "simulation_1/simData_", i, ".RDS") )
}
 

 
# Load necessary libraries
library(SingleCellExperiment)

n=1
for (i in seq.int(n) ){

# Load the RDS file
scdata <- readRDS(paste0('simulation_1/simData_',i,'.RDS'))

# Extract the counts data
counts_data <- assay(scdata, "counts")

# Extract row and column metadata
row_data <- as.data.frame(rowData(scdata))
col_data <- as.data.frame(colData(scdata))

# Save counts data to CSV
write.csv(as.data.frame(counts_data), paste0('simulation_1/counts_data_',i,'.csv'), row.names = FALSE)

# Save row and column metadata to CSV
write.csv(row_data, paste0('simulation_1/row_data_',i,'.csv'), row.names = FALSE)
write.csv(col_data, paste0('simulation_1/col_data_',i,'.csv'), row.names = FALSE)

}
