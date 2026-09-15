
library('arrow')
library('copykat')

setwd("/Users/sishirsubedi/Documents/projects/picasa/revision/copykat/")

samples <- c("raw","full", "common", "unique")

for (sample in samples) {
  message(paste0("Processing sample: ", sample))
  
  f <- paste0(sample, "_recons.parquet")
  df_raw <- read_parquet(f)

  rownames(df_raw) <- df_raw$'__index_level_0__'
  df_raw$'__index_level_0__' <- NULL
  exp.rawdata <- as.matrix(t(df_raw))

  # Subset to first 500 cells/samples
  # exp.rawdata <- exp.rawdata[, 1:500]

  copykat.raw <- copykat(
    rawmat = exp.rawdata,
    id.type = "S",
    ngene.chr = 5,
    win.size = 25,
    KS.cut = 0.1,
    sam.name = paste0(sample, "_data"),
    distance = "euclidean",
    plot.genes = FALSE,
    genome = "hg20",
    n.cores = 16
  )
}