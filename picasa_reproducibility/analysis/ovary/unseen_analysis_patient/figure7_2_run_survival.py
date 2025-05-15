import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')


import picasa
import anndata as ad
import pandas as pd


sample ='ovary'
df_latent = pd.read_csv('results/picasa_parameters_patient_by_factor.csv.gz',index_col=0)

####### get patient data #####################

df_pmeta = pd.read_csv('data/tcga_ovary_clinical.csv.gz')


df_pmeta['time'] = df_pmeta['days_to_death'].fillna(df_pmeta['days_to_last_follow_up'])
df_pmeta['event'] = (df_pmeta['vital_status'] == 'Dead').astype(int)

# check for missing values
print(df_pmeta[['time', 'event']].isnull().sum())
median_time = df_pmeta['time'].median()
df_pmeta['time'].fillna(median_time, inplace=True)
df_pmeta = df_pmeta.set_index('Unnamed: 0')


from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt


res_lrank = []    
res_kmf = []
df_latent = df_latent.loc[df_pmeta.index,:]
df = pd.concat([df_pmeta[['time', 'event']], df_latent], axis=1)

factors = df_latent.columns
num_factors = len(factors)

kmf = KaplanMeierFitter()

for idx, factor in enumerate(factors):
    median_value = df[factor].median()
    df['group'] = (df[factor] > median_value).astype(int)

    group_low = df[df['group'] == 0]
    group_high = df[df['group'] == 1]
    logrank_result = logrank_test(
        group_low['time'], group_high['time'], 
        event_observed_A=group_low['event'], 
        event_observed_B=group_high['event']
    )
    p_value = logrank_result.p_value
    chisq_stat = logrank_result.test_statistic
    group_high_mtime =  group_high['time'].mean()
    group_high_stdtime =  group_high['time'].std()
    group_low_mtime =  group_low['time'].mean()
    group_low_stdtime =  group_low['time'].std()
    res_lrank.append([factor,chisq_stat,p_value,
                group_high_mtime,
                group_high_stdtime,
                group_low_mtime,
                group_low_stdtime])
    
        
    for group, label in zip([0, 1], ['Low', 'High']):
        group_data = df[df['group'] == group]
        kmf.fit(group_data['time'], group_data['event'], label=label)
        for cdf_time,cdf in zip(kmf.cumulative_density_.index.values,kmf.cumulative_density_[label].values):
            res_kmf.append([factor,cdf_time,cdf,label])
            
        
pd.DataFrame(res_lrank,columns=['factor','chisq_stat','pval','group_high_mean_time','group_high_std_time','group_low_mean_time','group_low_std_time',]).to_csv('results/survival_analysis_logrank_result.csv.gz',compression='gzip')

pd.DataFrame(res_kmf,columns=['factor','time','cdf','label']).to_csv('results/survival_analysis_kmf_result.csv.gz',compression='gzip')