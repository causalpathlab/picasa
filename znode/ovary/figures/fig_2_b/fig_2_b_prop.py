import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os

sample = 'ovary'
wdir = 'znode/ovary/'
cdir = 'figures/fig_2_b/'
cancer = 'cancer'
df_umap = pd.read_csv(wdir+'results/df_umap_'+cancer+'.csv.gz')


#### patient proportion for each cluster 

from picasa.util.plots import get_colors
from plotnine import * 


for col in ['patient_id','treatment_phase','cell_type']:
    dfg = df_umap.groupby(['cluster',col]).count()['cell'].reset_index()
    cluster_sum = dict(dfg.groupby('cluster')['cell'].sum())
    dfg['ncount'] = [x/cluster_sum[y] for x,y in zip(dfg['cell'],dfg['cluster'])]

    dfg['cluster'] = ['c_'+str(x) for x in dfg['cluster']]

    nlabel = dfg[col].nunique()
    custom_palette = get_colors(28) 
    legend_size = 7

    p = (ggplot(data=dfg, mapping=aes(x='cluster', fill=col,weight='ncount')) +
    geom_bar(position='stack') +
    scale_color_manual(values=custom_palette)  +
    labs(x="Cluster", y=f"{col} distribution") +
    guides(color=guide_legend(override_aes={'size': legend_size})))

    p = p + theme(
        plot_background=element_rect(fill='white'),
        panel_background = element_rect(fill='white'),
        axis_text_x=element_text(rotation=45, hjust=1),
           axis_title_x=element_text(size=20),
           axis_title_y=element_text(size=20)
    )

    p.save(filename = wdir+cdir+'prop_'+col+'_'+cancer+'.pdf', height=10, width=15, units ='in', dpi=600)
    
    