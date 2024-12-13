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

SAMPLE = 'sim6'
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

def get_lisi(method,arr,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):    
    res = hm.compute_lisi(arr,df_meta,[batch_key,group_key])
    df_res = pd.DataFrame(res,columns=[method+'_'+batch_key,method+'_'+group_key])
    return df_res

def plot_lisi(df_lisi):
    
    df_main = df_lisi.melt()
    df_main.columns = ['batch','lisi']
    df_main['method'] = [x.split('_')[0] for x in df_main['batch']]
    df_main['type'] = [x.split('_')[1] for x in df_main['batch']]
        
        
    df_main = df_main[df_main['type']==constants.BATCH]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))  
    sns.boxplot(data=df_main, x="lisi", y="method", palette="Set2")
    sns.stripplot(data=df_main, x="lisi", y="method", color="grey", size=1, alpha=0.3, jitter=True)
    plt.xlabel("LISI Score", fontsize=12)
    plt.ylabel("Integration Method", fontsize=12)
    plt.title("Comparison of LISI Scores Across Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'benchmark_lisi.png'))
    plt.close()


##### NMI,ARI,Purity

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.preprocessing import StandardScaler 		
from sklearn.cluster import KMeans
from collections import Counter


def calc_scores(ct,cl):
	nmi =  normalized_mutual_info_score(ct,cl)
	ari = adjusted_rand_score(ct,cl)

	cluster_set = set(cl)
	total_correct = sum(max(Counter(ct[i] for i, cl in enumerate(cl) if cl == cluster).values()) 
						for cluster in cluster_set)
	purity = total_correct / len(ct)

	return nmi,ari,purity

def get_cluster(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):
    n_clust = df_meta[constants.GROUP].nunique()
    kmeans = KMeans(n_clusters=n_clust)
    kmeans.fit(df)
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

    plt.savefig(os.path.join(RESULTS_DIR,'benchmark_cluster.png'))
    plt.close()
    
###### ASW

from sklearn.metrics import silhouette_samples

def get_asw(method,df,df_meta,batch_key=constants.BATCH,group_key=constants.GROUP):
    
    asw_group = silhouette_samples(X=df,labels=df_meta[constants.GROUP])
    asw_group = np.mean(asw_group)

    asw_batch = silhouette_samples(X=df,labels=df_meta[constants.BATCH])
    asw_batch = np.mean(asw_batch)

    df_res = pd.DataFrame([[asw_group,asw_batch]],columns=[method+'_aws-'+constants.GROUP,method+'_aws-'+constants.BATCH])
    
    return df_res


def plot_asw(df_sil):
    df_sil = df_sil.T
    df_sil['method'] = [x.split('_')[0] for x in df_sil.index.values]
    df_sil['metric'] = [x.split('_')[1:][0] for x in df_sil.index.values]
    df_sil.rename(columns={0:'score'},inplace=True)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_sil, x='method', y='score', hue='metric', ci=None)

    plt.title("ASW evaluation")
    plt.xlabel("Method")
    plt.ylabel("Score")

    plt.savefig(os.path.join(RESULTS_DIR,'benchmark_asw.png'))
    plt.close()
         
############## eval 

df_meta = get_meta_data()

methods = ['combat','scvi','cellanova','harmony','pca','scanorama','liger','biolord']
df_lisi = pd.DataFrame()
df_clust = pd.DataFrame()
df_sil = pd.DataFrame()

for method in methods:
    df = pd.read_csv(os.path.join(RESULTS_DIR,'benchmark_'+method+'.csv.gz'))
    df.index = df.iloc[:,0]
    df = df.iloc[:,1:]
    df = df.loc[df_meta.index.values,:]

    df_lisi_res = get_lisi(method,df,df_meta)
    df_lisi = pd.concat([df_lisi,df_lisi_res],axis=1)

    df_clust_res = get_cluster(method,df,df_meta)
    df_clust = pd.concat([df_clust,df_clust_res],axis=1)

    df_sil_res = get_asw(method,df,df_meta)
    df_sil = pd.concat([df_sil,df_sil_res],axis=1)
    


### add picasa
picasa_adata = an.read_h5ad(os.path.join(RESULTS_DIR, 'picasa.h5ad'))

df_picasa_lisi = get_lisi('picasa',picasa_adata.obsm['common'],df_meta)
df_lisi = pd.concat([df_lisi,df_picasa_lisi],axis=1)

df_picasa_clust = get_cluster('picasa',picasa_adata.obsm['common'],df_meta)
df_clust = pd.concat([df_clust,df_picasa_clust],axis=1)

df_picasa_sil = get_asw('picasa',picasa_adata.obsm['common'],df_meta)
df_sil = pd.concat([df_sil,df_picasa_sil],axis=1)


#### plot
plot_lisi(df_lisi)
plot_clust(df_clust)
plot_asw(df_sil)




