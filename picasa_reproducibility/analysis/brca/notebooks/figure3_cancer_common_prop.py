import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from plotnine import * 


adata = ad.read_h5ad('data/figure3_cancer_common_selected_cells.h5ad')
####################################

dfg = adata.obs.groupby(['cluster','batch']).count()['batch_id'].reset_index()
cluster_sum = dict(dfg.groupby('cluster')['batch_id'].sum())
dfg['ncount'] = [x/cluster_sum[y] for x,y in zip(dfg['batch_id'],dfg['cluster'])]


nlabel = dfg['cluster'].nunique()
legend_size = 7

cluster_order = dfg['cluster'].unique()

dfg['cluster'] = pd.Categorical(dfg['cluster'], categories=cluster_order, ordered=True)

p = (ggplot(data=dfg, mapping=aes(x='cluster', fill='batch',weight='ncount')) +
geom_bar(position='stack') +
# scale_color_manual(values=custom_palette)  +
labs(x="Cluster", y="Patient distribution") +
guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'),
    axis_text_x=element_text(rotation=45, hjust=1,size=20),
        axis_title_x=element_text(size=20),
        axis_title_y=element_text(size=20)
)

p.save(filename = 'results/figure3_cancer_common_cluster_prop.pdf', height=10, width=15, units ='in', dpi=600)