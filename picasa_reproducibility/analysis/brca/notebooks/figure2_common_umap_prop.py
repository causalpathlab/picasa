import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from plotnine import * 


dfleiden = pd.read_csv('data/figure1_umap_coordinates.csv.gz',index_col=0)


dfg = dfleiden.groupby(['batch','celltype']).count()


dfg = dfleiden.groupby(['c_leiden','celltype']).count()['c_umap1'].reset_index()
celltype_sum = dict(dfg.groupby('c_leiden')['c_umap1'].sum())
dfg['ncount'] = [x/celltype_sum[y] for x,y in zip(dfg['c_umap1'],dfg['c_leiden'])]
dfg.sort_values(['c_leiden','ncount'],ascending=False,inplace=True)



dfg[ (dfg.celltype=='DC')]  

dfg.drop_duplicates(subset='c_leiden',inplace=True)

dfg = dfg.reset_index(drop=True).copy()

dfg[ (dfg.c_leiden==15)] 

## fix issue of very low DC cells
dfg.iloc[80,1] = 'DC'
dfg.iloc[80,2] = 507


dfg.drop_duplicates(subset='c_leiden',inplace=True)
dfg['p_label'] = ['Common'+str(x)+'/'+y for x,y in zip (dfg['c_leiden'],dfg['celltype'])]

dfg.sort_values(['c_umap1','celltype'],ascending=False,inplace=True)
dfg.drop_duplicates(subset='celltype',inplace=True)


dfleiden['p_label'] = ['Common'+str(x)+'/'+y for x,y in zip (dfleiden['c_leiden'],dfleiden['celltype'])]


dfleiden = dfleiden[dfleiden['p_label'].isin(dfg['p_label'])]
dfleiden['p_label'].value_counts()


dfg = dfleiden.groupby(['p_label','batch']).count()['c_umap1'].reset_index()
cluster_sum = dict(dfg.groupby('p_label')['c_umap1'].sum())
dfg['ncount'] = [x/cluster_sum[y] for x,y in zip(dfg['c_umap1'],dfg['p_label'])]



nlabel = dfg['p_label'].nunique()
legend_size = 7

p_label_order = dfg['p_label'].unique()

dfg['p_label'] = pd.Categorical(dfg['p_label'], categories=p_label_order, ordered=True)

from picasa.util import palette
color_palette = palette.colors_24

p = (ggplot(data=dfg, mapping=aes(x='p_label', fill='batch',weight='ncount')) +
geom_bar(position='stack') +
scale_fill_manual(values=color_palette)  +
labs(x="p_label", y="Patient distribution") +
guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'),
    axis_text_x=element_text(rotation=45, hjust=1,size=20),
        axis_title_x=element_text(size=20),
        axis_title_y=element_text(size=20)
)

p.save(filename = 'results/figure2_common_cluster_prop.pdf', height=10, width=15, units ='in', dpi=600)






df = pd.read_csv('data/figure1_umap_coordinates.csv.gz',index_col=0)


df = df.loc[dfleiden.index.values]

adata = sc.AnnData(df)
adata.obs['batch'] = df['batch'].astype('category')
adata.obs['celltype'] = df['celltype'].astype('category')
adata.obs['disease'] = df['disease'].astype('category')
adata.obs['c_leiden'] = df['c_leiden'].astype('category')
adata.obs['u_leiden'] = df['u_leiden'].astype('category')
adata.obs['b_leiden'] = df['b_leiden'].astype('category')
adata.obs['p_label'] = dfleiden['p_label'].astype('category')

umap_pairs = [('c_umap1', 'c_umap2'), ('u_umap1', 'u_umap2'), ('b_umap1', 'b_umap2')]




from picasa.util import palette
color_palette = palette.colors_24

cust_palette = sns.color_palette("tab20", len(adata.obs['p_label'].unique()))

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, (umap_x, umap_y) in enumerate(umap_pairs):
    sc.pl.scatter(adata, x=umap_x, y=umap_y, color='batch', palette= color_palette,ax=axes[i, 0], show=False)
    axes[i, 0].set_title(f"{umap_x[0]} (Batch)")

    sc.pl.scatter(adata, x=umap_x, y=umap_y,color = 'p_label',palette=cust_palette, ax=axes[i, 1], show=False)
    
    axes[i, 1].set_title(f"{umap_x[0]} (p_label)")

plt.tight_layout()
plt.savefig('results/figure2_latent_umaps_cluster.png')
