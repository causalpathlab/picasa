
from metrics import get_metrics_lisi,get_meta_data,get_metrics_others
import os 
import pandas as pd 
import glob
import sys 

SAMPLE = sys.argv[1]
# SAMPLE = 'lung'

WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'

DATA_DIR = os.path.join(WDIR, SAMPLE, 'results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')


methods = ['pca','bbknn','combat','harmony','scanorama','picasab','picasau','picasac','picasauc','scvi','liger']
# methods = ['picasab','picasau','picasac','picasauc']

for method in methods:
    print(method)
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'benchmark_'+method+'.csv.gz'))
    df.index = df.iloc[:,0]
    df = df.iloc[:,1:]

    df_meta = get_meta_data(SAMPLE, DATA_DIR)


    df = df.loc[df_meta.index.values,:]

    df_lisi_res = get_metrics_lisi(df,df_meta)
    df_others_res = get_metrics_others(df,df_meta)
    
    df_lisi_res['Method'] = method
    
    df_lisi_res = pd.concat([df_lisi_res,df_others_res],axis=1)

    df_lisi_res.to_csv(os.path.join(RESULTS_DIR, 'benchmark_'+method+'_scores.csv'),index=False) 
    
