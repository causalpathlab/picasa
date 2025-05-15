
source("Util.R")
theme_set(theme_classic() +
          theme(axis.text.y = element_text(size=8, hjust=1)) +
          theme(axis.text.x = element_text(size=8, angle = 90, hjust=1, vjust=.5)) +
          theme(legend.text = element_text(size=6)) +
          theme(legend.title = element_text(size=6)) +
          theme(legend.key.width = unit(0.2, "lines")) +
          theme(legend.key.height = unit(0.5, "lines")))

plot.cnv.corr <- function(data.name) {

    .file <- paste0("../figure3_cnv_celltype_corr_sp_copykat.csv.gz")

    .dt <- fread(.file,sep=",", header=T)

    if("treatment" %in% colnames(.dt)){
        .dt[, patient := paste0(patient,"_",treatment)]
    }

    if("subtype" %in% colnames(.dt)){
        .dt[, patient := paste0(patient,"_",subtype)]
    }

    .mat <-
        dcast(.dt, celltype ~ patient, value.var = "cnv_corr", fill=0) %>%
        column_to_rownames("celltype") %>%
        as.matrix()

    .ct.order <- names(row.order(.mat))

    .pt.order <- apply(.mat, 2, which.max) %>%
        order(decreasing=T) %>%
        (function(x) colnames(.mat)[x])

    .dt[, pt := factor(patient, .pt.order)]
    .dt[, ct := factor(celltype, .ct.order)]

    plt <-
        ggplot(.dt, aes(pt, ct, size=abs(cnv_corr), fill=pmin(pmax(cnv_corr, -.5), .5))) +
        xlab("patients") +
        ylab("cell types") +
        geom_point(pch=22, stroke=.1, colour="black") +
        scale_size_continuous("|R|", range=c(0, 5), breaks=seq(0,.5,.1)) +
        scale_fill_gradient2("CNV R\nmixed vs.\nunique",
                             low = "#998ec3",
                             mid = "#f7f7f7",
                             high = "#f1a340", midpoint = 0)

    if("treatment" %in% colnames(.dt)){
        plt <- plt + facet_grid(.~treatment, scales="free", space="free")
    }

    if("subtype" %in% colnames(.dt)){
        plt <- plt + facet_grid(.~subtype, scales="free", space="free")
    }

}

ggsave("cnv_cor_brca.pdf", plot=plot.cnv.corr("brca"), width=6, height=3.5)
