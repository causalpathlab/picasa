import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

############################
sample = 'brca' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/model_results/picasa.h5ad')

####################################

############ read original data as adata list


ddir = wdir+'/model_data/'
pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = ad.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=25:
		break

picasa_data = batch_map



## Focus on cancer cells

dmap = {
    'CID4495': 'TNBC',
    'CID44971': 'TNBC',
    'CID4471': 'ER+',
    'CID44991': 'TNBC',
    'CID4513': 'TNBC',
    'CID3586': 'HER2+',
    'CID4066': 'HER2+',
    'CID4290A': 'ER+',
    'CID4515': 'TNBC',
    'CID4530N': 'ER+',
    'CID3963': 'TNBC',
    'CID4535': 'ER+',
    'CID4067': 'ER+',
    'CID3921': 'HER2+',
    'CID4398': 'ER+',
    'CID4040': 'ER+',
    'CID45171': 'HER2+',
    'CID3838': 'HER2+',
    'CID44041': 'TNBC',
    'CID3948': 'ER+',
    'CID4523': 'TNBC'
}

picasa_adata.obs['disease']= [dmap[x] for x in picasa_adata.obs['batch']]


picasa_adata = picasa_adata[picasa_adata.obs['celltype']=='Malignant']

sc.pp.neighbors(picasa_adata,use_rep='unique')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata,resolution=0.1)

picasa_adata.obs['cluster'] = ['u_'+str(x) for x in picasa_adata.obs['leiden']]


###select clusters with >1k cells
# label_counts = picasa_adata.obs['cluster'].value_counts()
# filtered_labels = label_counts.index.values[:10]
# picasa_adata = picasa_adata[picasa_adata.obs['cluster'].isin(filtered_labels)]

sc.pl.umap(picasa_adata,color=['batch','celltype','cluster'])
plt.savefig('results/figure4_cancer_unique_umap.png')

sc.pl.umap(picasa_adata,color=['batch','celltype','disease'])
plt.savefig('results/figure4_cancer_unique_umap_disease.png')

### now save original data for the selected cells

df_main = pd.DataFrame()
for p_ad in picasa_data:
    df_main = pd.concat([df_main,picasa_data[p_ad].to_df()],axis=0)
    

picasa_adata.obs.index = [x.split('@')[0] for x in picasa_adata.obs.index.values]


df_main = df_main.loc[picasa_adata.obs.index.values]


adata = ad.AnnData(
    X=df_main.values,  
    obs=pd.DataFrame(index=df_main.index), 
    var=pd.DataFrame(index=df_main.columns)  
)

adata.obs = picasa_adata.obs

adata.write('data/figure4_cancer_unique_selected_cells.h5ad',compression='gzip')

