### chord diagram  ###

it_id = 'znode/pbmc/results/sc_context_module_'



plot_chordDiagram <- function(ct){

file = paste(it_id,ct,".csv.gz",sep="")
df = read.table(file, sep = ",", header=TRUE)
# df = df[order(df$topic),]
# df$topic = as.factor(df$topic)

print(dim(df))

library(reshape)
library("circlize")


dfm = melt(df)
colnames(dfm) = c('gene','context','value')
head(dfm)


dfm <- dfm[dfm$context != 'context6', ]
f = paste(it_id,ct,"_chorddiag.pdf",sep="")
pdf(f)


genes<- as.character(df[[1]])
othercol = structure(rep("grey", length(genes)), names = genes)
col_fun = colorRamp2(c(min(dfm$value), max(dfm$value)), c("white", "red"))


grid_col = c(
  "context1" = "#ff0000",  # Red
  "context2" = "#b3446c",  # Raspberry
  "context3" = "#0000ff",  # Blue
  "context4" = "#dcd300",  # Bright yellow
  "context5" = "#00ff00",  # Green
  "context6" = "#ff00ff",  # Magenta
  "context7" = "#882d17",  # Brick red
  "context8" = "#00ffff",  # Cyan
  "context9" = "#800080",  # Purple
  "context10" = "#8db600", # Yellow-green
  "context11" = "#ffa500", # Orange
  "context12" = "#000080", # Navy
  "context13" = "#ff69b4", # Hot pink
  "context14" = "#808080", # Gray
  "context15" = "#ffc0cb", # Pink
  "context16" = "#ffff00", # Yellow
  "context17" = "#008000", # Dark green
  "context18" = "#654522", # Milk chocolate
  "context19" = "#800000", # Maroon
  "context20" = "#000000", # Black
  "context21" = "#ff4500", # Orange-red
  "context22" = "#e25822", # Flame
  "context23" = "#2e8b57", # Sea green
  "context24" = "#2b3d26", # Pine green
  "context25" = "#4682b4",  # Steel blue
 othercol )

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

# cts = c('CD4+ T cell', 'Cytotoxic T cell', 'Dendritic cell', 'B cell', 'CD14+ monocyte', 'Natural killer cell', 'Megakaryocyte', 'CD16+ monocyte')
cts = c('CD4+ T cell', 'B cell', 'CD14+ monocyte')

for (ct in cts) {
    plot_chordDiagram(ct)
}

