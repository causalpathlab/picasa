# install.packages("BiocManager")
# BiocManager::install("fgsea")


library(fgsea)
library(data.table)
library(ggplot2)
library(dplyr)
source("Util.R")


run_gsea <- function(df,cutoff,msigdbr_df,outfile){

pathways = split(x = msigdbr_df$gene_symbol, f = msigdbr_df$gs_name)

selected_topics = colnames(df)

fr = data.frame()
for ( i in selected_topics ){
print(paste('processing',i))

sorder = order(df[,i], decreasing = TRUE)
ranks = df[sorder,i]
names(ranks) <- rownames(df)[sorder]
fgseaRes = fgsea(pathways = pathways, stats =ranks ,nperm=10000)
fgseaRes = fgseaRes[order(padj), ]
fgseaRes[,'context'] = i
# fgseaRes = fgseaRes[fgseaRes$padj<cutoff,]
fgseaRes=fgseaRes[1:5,]
fgseaRes = as.data.frame(fgseaRes)
fr <- rbind(fr,fgseaRes)
}

dfm=fr
dfm$padj = -log10(dfm$padj)
dfm = dcast(dfm,pathway~context,value.var="padj")
dfm[is.na(dfm)] <- 0
# dfm$topic = as.factor(dfm$topic)

rownames(dfm) = dfm$pathway
dfm$pathway = NULL

row_order = row.order(dfm)


# dfm_t = t(dfm)
# dfm_t = melt(dfm_t)
# colnames(dfm_t)=c('col','row','weight')
# col_order = col.order(dfm_t,rownames(dfm))

dfm = dfm[row_order,]
# dfm = dfm[,col_order]
# p <-   pheatmap(dfm,color =colorRampPalette(c("white", "tan2"))(100),fontsize_row=6,fontsize_col=8,cluster_rows=FALSE,cluster_cols=FALSE,column_names_side = c("top"),angle_col = c("45"))
dfm$pathway = rownames(dfm)

# dfm$pathway = sub("GOBP_", "", as.character(dfm$pathway))
# dfm$pathway = sub("GOCC_", "", as.character(dfm$pathway))
# dfm$pathway = sub("GOMF_", "", as.character(dfm$pathway))
dfm$pathway = substring( as.character(dfm$pathway),1,60)

dfm2 = melt(dfm)

dfmt = dfm2[dfm2$value > 0.0,]
dfmt_tab = dfmt%>% count(pathway)%>% filter(n<10)
dfm2 = dfm2[dfm2$pathway %in% dfmt_tab$pathway,]

p = ggplot(dfm2, aes(x = variable, y = pathway)) + 
geom_tile(aes(fill = value, width=0.8, height=0.8), size=0.1)+
scale_fill_gradient(low="white", high="blue")+
theme(legend.key=element_blank(), 
axis.text.x = element_text(colour = "black", size = 12, face = "bold", angle = 90, vjust = 0.3, hjust = 1), 
axis.text.y = element_text(colour = "black", face = "bold", size = 12), 
legend.text = element_text(size = 10, face ="bold", colour ="black"), 
legend.title = element_text(size = 12, face = "bold"), 
# panel.background = element_blank(), panel.border = element_rect(colour = "black", fill = NA, size = 1.2), 
legend.position = "right") +
labs(fill = "LogAdjPval")

ggsave(outfile,p,width = 15, height = 10,dpi=600)

}




# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP")
# tag='cp2'
# outfile=paste('12_gsea_cp_',tag,'.pdf',sep='')
# run_gsea(df,cutoff,msigdbr_df,outfile)

# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP:KEGG")
# tag='KEGG'
# outfile=paste('12_gsea_cp_',tag,'.pdf',sep='')
# run_gsea(df,cutoff,msigdbr_df,outfile)

# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP:REACTOME")
# tag='REACTOME'
# outfile=paste('12_gsea_cp_',tag,'.pdf',sep='')
# run_gsea(df,cutoff,msigdbr_df,outfile)

# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP:BIOCARTA")
# tag='BIOCARTA'
# outfile=paste('12_gsea_cp_',tag,'.pdf',sep='')
# run_gsea(df,cutoff,msigdbr_df,outfile)

# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP:PID")
# tag='PID'
# outfile=paste('12_gsea_cp_',tag,'.pdf',sep='')
# run_gsea(df,cutoff,msigdbr_df,outfile)





# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C7",subcategory = "IMMUNESIGDB")
msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C2",subcategory = "CP:BIOCARTA")
# msigdbr_df <- msigdbr::msigdbr(species = "human", category = "C8")

cts = c('Mac', 'T', 'NK', 'B', 'ATII', 'EC', 'Fib')


for (ct in cts) {
    file = paste("znode/lung/results/sc_context_module_gset_",ct,".csv.gz",sep="")
    df = read.table(file, sep = ",", header=TRUE,row.names=1)

    cutoff = 0.05
    outfile=paste('znode/lung/results/sc_context_gsea_',ct,'.pdf',sep='')
    run_gsea(df,cutoff,msigdbr_df,outfile)

}

