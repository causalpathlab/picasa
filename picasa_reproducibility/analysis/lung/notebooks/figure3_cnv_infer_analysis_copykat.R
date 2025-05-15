
library('arrow')
library('copykat')

raw_files = list.files(path = ".", pattern = "^raw_data_")

for (f in raw_files){
    patient_name = strsplit(f,"_")[[1]][3]
    celltype_name = sub("\\.parquet$", "",strsplit(f,"_")[[1]][4])
    print(paste(patient_name,celltype_name))
    ptct_name = paste0("raw_",patient_name,"_",celltype_name,sep="_")

    df_raw = read_parquet(f)
    rownames(df_raw) <- df_raw$'__index_level_0__'
    df_raw$'__index_level_0__' <- NULL
    exp.rawdata <- as.matrix(t(df_raw))

    copykat.raw <- copykat(rawmat=exp.rawdata, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name=ptct_name, distance="euclidean", plot.genes="TRUE", genome="hg20",n.cores=32)

}

recons_files = list.files(path = ".", pattern = "^recons_data_")

for (f in recons_files){
    patient_name = strsplit(f,"_")[[1]][3]
    celltype_name = sub("\\.parquet$", "",strsplit(f,"_")[[1]][4])
    print(paste(patient_name,celltype_name))
    ptct_name = paste0("recons_",patient_name,"_",celltype_name,sep="_")

    df_raw = read_parquet(f)
    rownames(df_raw) <- df_raw$'__index_level_0__'
    df_raw$'__index_level_0__' <- NULL
    exp.rawdata <- as.matrix(t(df_raw))
    
    copykat.raw <- copykat(rawmat=exp.rawdata, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name=ptct_name, distance="euclidean", plot.genes="TRUE", genome="hg20",n.cores=32)

}

