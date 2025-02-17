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


sample ='lung'
adata = an.read_h5ad('results/picasa.h5ad')

####### get patient data #####################

df_pmeta = pd.read_csv('data/tcga_lung_clinical.csv.gz')


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

selected_topics = []
def surv_plot(tag):
    df_latent = pd.DataFrame(adata.obsm[tag], index=adata.obs_names)
    df_latent = df_latent.loc[:, df_latent.median() != 0]

    df_latent.columns = ['k'+str(x)for x in df_latent.columns]
    
    selected_topics = ['k5', 'k11', 'k47']
    df_latent = df_latent.loc[:,selected_topics]
    
    df_latent = df_latent.loc[df_pmeta.index,:]
    df = pd.concat([df_pmeta[['time', 'event']], df_latent], axis=1)

    factors = df_latent.columns
    num_factors = len(factors)

    rows = (num_factors + 2) // 3 
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  
    axes = axes.flatten()  
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

        if p_value > 0.01:
            continue
        selected_topics.append(factor)
        
        for group, label in zip([0, 1], ['Low', 'High']):
            group_data = df[df['group'] == group]
            kmf.fit(group_data['time'], group_data['event'], label=label)
            kmf.plot_survival_function(ax=axes[idx],ci_show=False, linewidth=1,marker='o',markersize=5)

        axes[idx].set_title(f"Survival Curve for {factor}\nP-value: {p_value:.4e}")
        axes[idx].set_xlabel("Time (days)")
        axes[idx].set_ylabel("Survival Probability")
        axes[idx].legend()

    # remove empty plots
    for ax in axes[len(factors):]:
        ax.remove()

    plt.tight_layout()
    plt.savefig('results/survival_all_factors_' + tag + '.pdf')

surv_plot('picasa')
print(selected_topics)

