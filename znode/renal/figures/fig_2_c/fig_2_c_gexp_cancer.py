import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os

sample = 'ovary'
wdir = 'znode/ovary/'
cdir = 'figures/fig_2_c/'

df_umap = pd.read_csv(wdir+'results/df_umap_cancer.csv.gz')


### get raw data 
directory = wdir+'/data'
pattern = 'ovary_*.h5ad'
file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break

file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,
	wdir)

df_main = pd.DataFrame()
for ad in picasa_object.adata_keys:
    df_main = pd.concat([df_main,picasa_object.data.adata_list[ad].to_df()],axis=0)


df_umap['cluster'] = ['c_'+str(x) for x in df_umap['cluster'].values]
df_umap.sort_values('cluster',inplace=True)

df_main = df_main.loc[df_umap['cell'].values,:]

df_main.index = [y+'@'+x for x,y in zip(df_main.index.values,df_umap['cluster'].values)]

import anndata as ad
import scanpy as sc
adata = ad.AnnData(df_main)

adata.obs['cluster'] = [x.split('@')[0] for x in adata.obs.index.values]

sc.tl.rank_genes_groups(adata, "cluster", method="wilcoxon")

df_results = pd.DataFrame(adata.uns['rank_genes_groups']['names'])

top_genes  = []
top_num = 5
for col in df_results.columns:
    for gene in df_results[col].values[:top_num]:
        if gene not in top_genes: 
            top_genes.append(gene)

###### top genes 
df_main = df_main.loc[:,top_genes]

df_main['cluster'] = [x.split('@')[0] for x in df_main.index.values]

sample_num= 7000
df_sampled = df_main.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_num)))

df_sampled = df_sampled.loc[:,df_sampled.columns[:-1]]



df_sampled = df_sampled.T

df_sampled.columns = [x.split('@')[0] for x in df_sampled.columns]


sns.clustermap(np.log1p(df_sampled.T),cmap="viridis",row_cluster=False)
plt.savefig(wdir+cdir+'hmap_cancer.png')
plt.close()

######marker genes 

genes = """
DST, EGFR, EPHB2, ITGA6, PIK3CD, PLEC, SFN, SMAD3, CLTC, EIF2AK2, ITGA2, KPNA1, STAT1, BCAR1, CCND1, COL1A2, ITCH, MMP1, MMP12, PML, TNC, CDK6, FADD, FAS, MX1, E2F3, BIRC2, CAPN2, VASP, MMP9, DFFA, KPNB1, ADAMTS5, MMP10, AP2B1, ASAP1, SDCBP, CYP1A1, CYP1B1, CYP3A5, MAFG, SLC7A11, IGF2,
B3GNT7, B3GNT8, GCNT3, MUC15, MUC4, MUC5B, ST3GAL4, LCN2, S100A8, S100A9, ALOX5, AOC1, B2M, C3, CEACAM1, CEACAM6, MGST1, OSTF1, PPBP, S100P, SLPI, TMEM173, DSC2, EVPL, IVL, KRT13, KRT4, KRT6A, KRT6C, HRASLS2, LPCAT4, PLBD1, RARRES3, CDKN2B, STEAP4, TGFB1, CFB, CCL28, CX3CL1, CXCL10,
AURKA, AURKB, BIRC5, BUB1, BUB1B, CCNA2, CCNB1, CCNB2, CDC20, CDC25C, CDC45, CDC6, CDCA8, CDK1, CDT1, CENPA, CENPE, CENPF, CENPH, CENPK, CENPM, CENPN, CENPU, CKAP5, DHFR, E2F1, H2AFV, H2AFZ, HAUS1, HIST1H2BH, HIST1H3G, HIST1H4C, HIST2H2AC, KIF18A, KIF20A, KIF23, KIF2C, LIG1, LMNB1, MAD2L1, MCM10, MCM2, MCM3, MCM4, MCM5, MCM6, MCM7, MCM8, NDC80, NEK2, NUF2, ORC6, PCNA, PLK1, PLK4, POLD1, POLD2, PRIM1, PTTG1, RFC2, RFC3, RFC4, RRM2, SGO1, SGO2, SKA1, SKA2, SKP2, SPC24, SPC25, TK1, TMPO, TPX2, UBE2C, VRK1, ZWILCH, ZWINT, BRCA1, CHEK1, CLSPN, RHNO1, RMI2, FANCG, HMGB2, KIF4A, RRM1, TYMS, RACGAP1, CBX5, BRCA2, FANCD2, RAD51, RAD51AP1, FANCB, FANCI, UBE2T, KIF11, KIF15, KIF20B, KIF22, KIFC1, HIST1H1A, HIST1H1B, GGH, DTYMK, AKR1B1,
CEBPB, CEBPD, FOS, IL6, JUN, JUNB, MCL1, MYC, SOCS3, ATF3, DUSP1, EGR1, FOSB, CEBPA, DDIT3, EGR2, CDKN1A, GADD45B, TNF, HES1, HBEGF, BCL6, NR4A1, DUSP6, GADD45G, ID2, NFKBIA, PLK3, SNAI2, CREB5, HLA-G, HIST1H2BC, HIST1H2BG, CALML3, SNAI1,
CCL20, CXCL1, IL1R2, IL1RN, SEC11C, SPCS3, BIK, BIRC3, CDKN2A, DAB2, GJB2,
CTSL, HLA-DMA, HLA-DMB, HLA-DOA, HLA-DPA1, HLA-DPB1, HLA- DQA1, HLA-DQA2, HLA-DRA, HLA-DRB5, HSP90AB1, HSPA2, HSPA4, HSPA5, HSPA6, PDIA3, CPE, CTSF, C1R, CXADR, SFTPD, TUBB2A, TUBB6, FOLR1, GAS6, NPC2, RUNX1, TGFBR2, GNAI1, RBPJ, IL11RA,HIST1H2AE, DNAJC3, FBXO2, HERPUD1, HSP90B1, SKP1, UBE2D1, UBQLN1, NECTIN2, SDC2, CCND3, DNAJB9, ERP27, SIAH1, PPP2CA, BAG3, HSPA12A, BMP2, LATS2, PPP2R2B, PRKCI, WNT6, IFITM1, IFITM2, CRMP1, DPYSL3, PLXNA2, MYL9, TNNC1, TNNI1, TNNT2, TLE1, TLE4, APP, PELI2, VAMP2, DNAJA4, PLAT, ACADVL, TSPYL2, NR1D2, PURA, GATA6, TWIST1, RING1, BAMBI, ATG101, RAPGEF2, NR4A3, SOX4, SLC22A18,
GBP4, IFI27, IFI35, IFIT1, ISG15, OAS1, OAS2, STAT2, TRIM22, TAP1, ITGAV,
BMS1, DCAF13, MPHOSPH10, PNO1, ACIN1, ROCK1, TJP1, ESCO1, WAPL, EIF1AX, PNN, RNPS1, DYNC1LI1, BDP1, CRCP, PAFAH1B1, PRPF40A, SFSWAP, KMT2A, SETD2,
AIMP1, EIF3I, MRPL13, MRPL15, MRPL16, MRPL2, MRPL28, MRPL3, MRPS7, RPN2, SRP9, TSFM, PSMA4, PSMB5, PSMB6, PSMB7, PSMC5, PSMD8, ACTL6A, BANF1, PPIA, XRCC6, VDAC3, FH, MDH2, SDHB, TPI1, ECHS1, HSD17B10, POLR2G, NDUFB6, HNRNPA2B1, HNRNPA3, SNRPD3, PRDX3, PTMA, CCT7, VBP1, COPE,
GLO1, IDH3B, NDUFA10, NDUFB5, PDHA1, SUCLG1, TRAP1, UQCRC1, UQCRC2, MDH1, ALDH7A1, DECR1, ECH1, ECI2, PCCB, SHMT1, HACD3, PTGR1, MGST2, PRDX6
"""
gene_list = [gene.strip() for gene in genes.split(",")]

sg = []
for g in gene_list:
    if g not in sg:
        sg.append(g)
        
marker = [ x for x in sg if x in df_main.columns]
df_main = df_main.loc[:,marker]

df_main['cluster'] = [x.split('@')[0] for x in df_main.index.values]

sample_num= 400
df_sampled = df_main.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_num)))

df_sampled = df_sampled.loc[:,df_sampled.columns[:-1]]



df_sampled = df_sampled.T

df_sampled.columns = [x.split('@')[0] for x in df_sampled.columns]


sns.clustermap(np.log1p(df_sampled.T),cmap="viridis",row_cluster=False)
plt.savefig(wdir+cdir+'hmap_marker_cancer.png')
plt.close()


from picasa.util.plots import plot_marker_genes

umap_coords = df_umap[['umap1','umap2']].values
marker = ["IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14",
    "LGALS3", "S100A8", "GNLY", "NKG7", "KLRB1",
    "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP"]

plot_marker_genes(wdir+cdir,df_main.iloc[:,:-1],umap_coords,marker,nr=4,nc=5)