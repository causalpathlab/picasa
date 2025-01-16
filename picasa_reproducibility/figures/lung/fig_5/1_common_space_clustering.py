import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')


############################
# sample = sys.argv[1] 
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/results/picasa.h5ad')
wdir = wdir + '/fig_2/'

####################################


## Focus on cancer cells

sc.pp.neighbors(picasa_adata,use_rep='common')
sc.tl.umap(picasa_adata)
sc.tl.leiden(picasa_adata)

picasa_adata.obs['cluster'] = ['c_'+str(x) for x in picasa_adata.obs['leiden']]

picasa_adata = picasa_adata[picasa_adata.obs['celltype']=='Malignant']

###select clusters with >1k cells
label_counts = picasa_adata.obs['cluster'].value_counts()
filtered_labels = label_counts.index.values[:10]

picasa_adata = picasa_adata[picasa_adata.obs['cluster'].isin(filtered_labels)]

sc.pl.umap(picasa_adata,color=['batch','celltype','cluster'])
plt.savefig(wdir+'/results/selected_cluster_umap_org_space.png')

picasa_adata.obs.to_csv(wdir+'/results/common_space_selected_cells.csv.gz',compression='gzip')


'''
Here, we use common representation space to cluster cells using leiden algorithm
with default settings. We select malignant cell labels and top 10 clusters by total number 
of cells. Then we draw umap of those selected clusters onto original common space with cluster labels.
'''