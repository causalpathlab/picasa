
import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from anndata import AnnData
import picasa
from scipy.sparse import csr_matrix 


############# data prep for simulation 

dfb = pd.read_csv('MN_CyTOF_Bpanel_matrix.csv')
dft = pd.read_csv('MN_CyTOF_Tpanel_matrix.csv')


dfsp = pd.read_csv('mibi_subexpression.csv')
dfsp_loc = pd.read_csv('mibi_sublocation.csv')

sample_n = 3000
dfb = dfb.sample(sample_n)
dft = dft.sample(sample_n)
dfsc = pd.merge(dft,dfb,on='File.Name',how='left')

rna = ad.read_h5ad('brca_sc.h5ad')
spatial = ad.read_h5ad('brca_sp.h5ad')

# pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})
# picasa.pp.common_features(pico.data.adata_list)

# rna_genes = rna.var.index.values[rna.uns['selected_genes']]



dfl = pd.read_csv('brca_scRNA_celllabels_subset.txt',sep='\t')
dfl.columns = ['cell','celltype']

rna.obs['celltype'] = pd.merge(rna.obs,dfl,left_index=True, right_on='cell',how='left')['celltype'].values

sel_ct_map = {
'T cells CD8':'Lymphoid', 
'T cells CD4':'Lymphoid', 
'Monocytes and Macrophages':'Myeloid',
'Epithelial cells':'Epithelial', 
'NK cells':'Lymphoid', 
'Fibroblasts':'Fibroblast', 
'Endothelial cells':'Endothelial',
'B cells':'Lymphoid', 
'PVL':'PVL', 
'PCs':'PCs', 
'Dendritic cells':'Dendritic'
}

sel_ct = ['Endothelial', 'Fibroblast', 'Lymphoid','Myeloid', 'Epithelial']

rna.obs['celltype'] = [sel_ct_map[x] for x in rna.obs['celltype']]
rna = rna[rna.obs['celltype'].isin(sel_ct)] 



rna.write('sim_sc.h5ad',compression='gzip')

spatial.write('sim_sp.h5ad',compression='gzip')

# ##################################


### this is for dice dataset

# import sys
# sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

# import os
# import glob
# import matplotlib.pylab as plt
# import anndata as ad
# import scanpy as sc
# import pandas as pd
# import numpy as np
# from anndata import AnnData
# import picasa
# from scipy.sparse import csr_matrix 

# def get_bulkdata(bulk_path):
		
# 	files = []
# 	for file in glob.glob(bulk_path):
# 		files.append(file)
	

# 	dfall = pd.DataFrame()
# 	cts = []
# 	for i,f in enumerate(files):
# 		print('processing...'+str(f))
# 		df = pd.read_csv(f)
# 		df = df[df['Additional_annotations'].str.contains('protein_coding')].reset_index(drop=True)
# 		df = df.drop(columns=['Additional_annotations'])
		
# 		ct = os.path.basename(f).split('.')[0].replace('_TPM','')
# 		cols = [str(x)+'_'+ct for x in range(df.shape[1]-2)]
# 		df.columns = ['gene','length'] + cols
		
# 		if i == 0:
# 			dfall = df
# 		else:
# 			dfall = pd.merge(dfall,df,on=['gene','length'],how='outer')
# 		cts.append(ct)
# 	return dfall,cts

# seed = 123
# from picasa.util.hvgenes import select_hvgenes

# np.random.seed(seed)

# ## dice bulk data
# bulk_path='/data/sishir/database/dice_immune_bulkrna/*.csv'
# df,cts = get_bulkdata(bulk_path)
# nz_cutoff = 10
# df= df[df.iloc[:,2:].sum(1)>nz_cutoff].reset_index(drop=True)
# genes = df['gene'].values
# glens = df['length'].values
# df = df.drop(columns=['gene','length'])


# ## scale up
# x_sum = df.values.sum(0)
# df = (df/x_sum)*10000

# df = df.T  
# hvgs = df.columns.values[select_hvgenes(df.to_numpy(),gene_var_z=2.0)]

# df = df.loc[:,hvgs]
# df.columns = genes[hvgs]

# adata_rna = ad.AnnData(X=df.values)
# adata_rna.obs.index = df.index.values
# adata_rna.var.index = df.columns.values
# adata_rna.obs['celltype'] = [''.join(x.split('_')[1:]) for x in df.index.values]

# adata_rna.write('For_sim_sc.h5ad',compression='gzip')
