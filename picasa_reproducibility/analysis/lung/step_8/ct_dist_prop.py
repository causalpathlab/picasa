import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd 
from plotnine import * 

sample ='lung'
pp = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'
wdir = pp + sample + '/fig_8/'




df_o = pd.read_csv('results/cnv_prop_orig.csv.gz')
df_r = pd.read_csv('results/cnv_prop_recons.csv.gz')

df = pd.DataFrame()
df['omean'] = df_o.groupby('celltype')['ncount'].max()
df['rmean'] = df_r.groupby('celltype')['ncount'].max()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df.reset_index(inplace=True)
dfm = pd.melt(df[['celltype','omean','rmean']],id_vars='celltype')

plt.figure(figsize=(12, 6))
sns.barplot(data=dfm, x="celltype", y="value", hue="variable", palette="Set2")
plt.xticks(rotation=45, ha="right")
plt.title("Side-by-Side Bar Plot of Omean and Rmean")
plt.xlabel("Celltype")
plt.ylabel("Value")
plt.legend(title="Variable")
plt.tight_layout()
plt.savefig(wdir+'results/cnv_prop_all.pdf')