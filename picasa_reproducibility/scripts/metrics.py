import os
import anndata as an
import pandas as pd
import numpy as np
import constants 


def get_meta_data(SAMPLE,DATA_DIR):
	picasa_adata = an.read_h5ad(os.path.join(DATA_DIR, 'picasa.h5ad'))
	df_meta = picasa_adata.obs.copy()
	if SAMPLE == 'lung':
		df_meta.index = ['@'.join(x.split('@')[:2]) for x in df_meta.index.values]
	else:
		df_meta.index = [x.split('@')[0] for x in df_meta.index.values]
	return df_meta



##### LISI

import harmonypy as hm 
def get_metrics_hm_batch(df,df_meta,batch_key=constants.BATCH):
	
	lisi_res = hm.compute_lisi(df,df_meta,[batch_key])
	return np.mean(lisi_res),np.std(lisi_res)
   

def get_metrics_hm_group(df,df_meta,group_key=constants.GROUP):
	

	lisi_res = hm.compute_lisi(df,df_meta,[group_key])
	return np.mean(lisi_res),np.std(lisi_res)

def get_metrics(df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    

	avg_res = []
	
	ilisi_res_mean,ilisi_res_std = get_metrics_hm_batch(df,df_meta,batch_key)
 
	clisi_res_mean,clisi_res_std = get_metrics_hm_group(df,df_meta,group_key)
 
	avg_res.append([ilisi_res_mean,ilisi_res_std,clisi_res_mean,clisi_res_std])

	df_res = pd.DataFrame(avg_res,columns=['ilisi_mean','ilisi_std','clisi_mean','clisi_std'])
	
	return df_res.round(3)
