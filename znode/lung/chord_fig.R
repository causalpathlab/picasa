### chord diagram  ###

it_id = 'znode/pancreas/results/sc_context_module_'



plot_chordDiagram <- function(ct){

file = paste(it_id,ct,".csv.gz",sep="")
df = read.table(file, sep = ",", header=TRUE)
# df = df[order(df$topic),]
# df$topic = as.factor(df$topic)

print(dim(df))
print(head(df))

library(reshape)
library("circlize")


dfm = melt(df)

print(head(dfm))

colnames(dfm) = c('gene','context','value')
head(dfm)

f = paste(it_id,ct,"_chorddiag.pdf",sep="")
pdf(f)


genes<- as.character(df[[1]])
othercol = structure(rep("grey", length(genes)), names = genes)
grid_col = c(othercol )
col_fun = colorRamp2(c(min(dfm$value), max(dfm$value)), c("blue", "red"))

# Plot chord diagram
chordDiagram(dfm,
             grid.col=grid_col,
            #  col = grid_col[as.character(dfm[[1]])],
             col = col_fun(dfm$value),
             annotationTrack = c("grid"), 
             preAllocateTracks = 1
             )

# highlight.sector(dfm$gene,
#                  track.index = 1, col = "white",
#                  text = "Genes", cex = 1, text.col = "black", 
#                  niceFacing = TRUE, font=2)

highlight.sector(dfm$context,
                 track.index = 1, col = "white",
                 text = "Context", cex = 1, text.col = "black", 
                 niceFacing = TRUE, font=2)

circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  circos.text(mean(xlim), ylim[1] + .1, sector.name, facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5),cex=0.4)
  circos.axis(h = "top",labels=NULL)
}, bg.border = NA)
dev.off()
circos.clear()
}

cts = c('Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib')

for (ct in cts) {
    plot_chordDiagram(ct)
}

