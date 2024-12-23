import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as an
import pandas as pd
import scanpy as sc
import numpy as np

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SAMPLE = sys.argv[1] 
# WDIR = sys.argv[2]

SAMPLE = 'pancreas' 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_meta_data():
    picasa_adata = an.read_h5ad(os.path.join(RESULTS_DIR, 'picasa.h5ad'))
    df_meta = picasa_adata.obs.copy()
    df_meta.index = [x.split('@')[0] for x in df_meta.index.values]
    return df_meta


##### LISI
import harmonypy as hm

def get_lisi(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    
    # res = hm.compute_lisi(df,df_meta,[batch_key,group_key])
    # df_res = pd.DataFrame(res,columns=[method+'_'+batch_key,method+'_'+group_key])
    # return df_res

    res = []
    for group in df_meta[constants.GROUP].unique():
        indices = df_meta[df_meta[constants.GROUP] == group].index.values
        
        ## need minimum sample >= 100 for compute_lisi to work
        
        if len(indices) < 100:
            continue
        
        res_lisi = hm.compute_lisi(df.loc[indices],df_meta.loc[indices],[batch_key])
        # min_lisi = np.min(res_lisi)
        # max_lisi = np.max(res_lisi)
        # normalized_lisi = (res_lisi - min_lisi) / (max_lisi - min_lisi)
        # res.append([method+'_'+group, np.mean(normalized_lisi)])
        res.append([method+'_'+group, np.mean(res_lisi)])

    df_res = pd.DataFrame(res,columns=['lisi','score'])
    return df_res

    
def plot_lisi(df_lisi):
    
    df_lisi['method'] = [x.split('_')[0] for x in df_lisi['lisi']]
        
        
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_lisi, x="method", y="score", palette="Set2")
    sns.stripplot(data=df_lisi, x="method", y="score", color="grey", size=1, alpha=0.3, jitter=True)
    
    plt.title("LISI evaluation by group")
    plt.xlabel("Method")
    plt.ylabel("Score")

    plt.savefig(os.path.join(RESULTS_DIR,'benchmark_plot_lisi_group.png'))
    plt.close()
    


##### NMI,ARI,Purity

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.preprocessing import StandardScaler 		
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import StandardScaler

def calc_scores(ct,cl):
	nmi =  normalized_mutual_info_score(ct,cl)
	ari = adjusted_rand_score(ct,cl)

	cluster_set = set(cl)
	total_correct = sum(max(Counter(ct[i] for i, cl in enumerate(cl) if cl == cluster).values()) 
						for cluster in cluster_set)
	purity = total_correct / len(ct)

	return nmi,ari,purity

def get_cluster(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):
    

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    n_clust = df_meta[constants.GROUP].nunique()
    kmeans = KMeans(n_clusters=n_clust)
    kmeans.fit(df_scaled)
    klabels = kmeans.labels_
    nmi,ari,purity = calc_scores(df_meta[constants.GROUP],klabels)
    
    df_res = pd.DataFrame([[nmi,ari,purity]],columns=[method+'_nmi',method+'_ari',method+'_purity'])
    
    return df_res


def plot_clust(df_clust):
    df_clust = df_clust.T
    df_clust['method'] = [x.split('_')[0] for x in df_clust.index.values]
    df_clust['metric'] = [x.split('_')[1] for x in df_clust.index.values]
    df_clust.rename(columns={0:'score'},inplace=True)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_clust, x='method', y='score', hue='metric', ci=None)

    plt.title("Cluster evaluation")
    plt.xlabel("Method")
    plt.ylabel("Score")

    plt.savefig(os.path.join(RESULTS_DIR,'benchmark_plot_cluster.png'))
    plt.close()
    
###### ASW

from sklearn.metrics import silhouette_score,silhouette_samples

def get_asw(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):
    
    res = []
    for group in df_meta[constants.GROUP].unique():
        indices = df_meta[df_meta[constants.GROUP] == group].index.values
        batch_labels = df_meta[df_meta[constants.GROUP] == group][constants.BATCH].values
        sil = silhouette_samples(X=df.loc[indices],labels=batch_labels)
        sil = [abs(i) for i in sil]
        res.append([method+'_asw-'+group, np.mean([1 - i for i in sil])])

    df_res = pd.DataFrame(res,columns=['aws','score'])
    return df_res


def plot_asw(df_sil):
    df_sil['method'] = [x.split('_')[0] for x in df_sil['aws']]
    df_sil['metric'] = [x.split('_')[1] for x in df_sil['aws']]
    

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_sil, x="method", y="score", palette="Set2")
    sns.stripplot(data=df_sil, x="method", y="score", color="grey", size=1, alpha=0.3, jitter=True)
    
    plt.title("ASW evaluation by group")
    plt.xlabel("Method")
    plt.ylabel("Score")

    plt.savefig(os.path.join(RESULTS_DIR,'benchmark_plot_asw_group.png'))
    plt.close()
    
    
############## eval 

df_meta = get_meta_data()

methods = ['pca','combat','harmony', 'scanorama','liger','scvi','cellanova']

df_lisi = pd.DataFrame()
df_clust = pd.DataFrame()
df_sil = pd.DataFrame()

for method in methods:
    print('eval '+method)
    df = pd.read_csv(os.path.join(RESULTS_DIR,'benchmark_'+method+'.csv.gz'))
    df.index = df.iloc[:,0]
    df = df.iloc[:,1:]
    df = df.loc[df_meta.index.values,:]

    df_lisi_res = get_lisi(method,df,df_meta)
    df_lisi = pd.concat([df_lisi,df_lisi_res],axis=0)

    # df_clust_res = get_cluster(method,df,df_meta)
    # df_clust = pd.concat([df_clust,df_clust_res],axis=1)

    df_sil_res = get_asw(method,df,df_meta)
    df_sil = pd.concat([df_sil,df_sil_res],axis=0)
    


### add picasa
picasa_adata = an.read_h5ad(os.path.join(RESULTS_DIR, 'picasa.h5ad'))
df_p = picasa_adata.obsm['common'].copy()
df_p.index = [x.split('@')[0] for x in df_p.index.values]

df_picasa_lisi = get_lisi('picasa',df_p,df_meta)
df_lisi = pd.concat([df_lisi,df_picasa_lisi],axis=0)

# df_picasa_clust = get_cluster('picasa',picasa_adata.obsm['common'],df_meta)
# df_clust = pd.concat([df_clust,df_picasa_clust],axis=1)

df_picasa_sil = get_asw('picasa',df_p,df_meta)
df_sil = pd.concat([df_sil,df_picasa_sil],axis=0)


#### plot
plot_lisi(df_lisi)
# plot_clust(df_clust)
# plot_asw(df_sil)




