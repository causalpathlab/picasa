import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np




import umap 
from picasa.util.plots import plot_umap_df

sample = 'ovary'
wdir = 'znode/ovary/'
 

df_u = pd.read_csv(wdir+'results/df_u.csv.gz',index_col=0)

dfl = pd.read_csv(wdir+'data/ovary_label.csv.gz')
dfl = dfl[['index','cell','patient_id','cell_type','treatment_phase']]
dfl.columns = ['index','cell','batch','celltype','treatment_phase']
dfl.cell = [x+'@'+y for x,y in zip(dfl['cell'],dfl['batch'])]


for sel_patient in dfl['batch'].value_counts().index.values:
	sel_patient_cells = dfl[dfl['batch']==sel_patient]['cell'].values

	df_u_c = df_u.loc[sel_patient_cells]

	umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.3,n_neighbors=20,metric='cosine').fit(df_u_c)
	df_umap= pd.DataFrame()
	df_umap['cell'] = df_u_c.index.values
	df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]


	df_umap['celltype'] = pd.merge(df_umap,dfl,on='cell',how='left')['celltype'].values
	plot_umap_df(df_umap,'celltype',wdir+'results/nn_attncl_lat_unq_'+sel_patient,pt_size=1.0,ftype='png')

	df_umap['batch'] = pd.merge(df_umap,dfl,on='cell',how='left')['batch'].values
	plot_umap_df(df_umap,'batch',wdir+'results/nn_attncl_lat_unq_batch_'+sel_patient,pt_size=1.0,ftype='png')

	df_umap['treatment_phase'] = pd.merge(df_umap,dfl,on='cell',how='left')['treatment_phase'].values
	plot_umap_df(df_umap,'treatment_phase',wdir+'results/nn_attncl_lat_unq_batch_'+sel_patient,pt_size=1.0,ftype='png')


