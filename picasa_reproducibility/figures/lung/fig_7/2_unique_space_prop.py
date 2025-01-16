import sys
import scanpy as sc
import matplotlib.pylab as plt
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/')

dark_colors = [
    "#1B1B1B",  # Dark Gray
    "#2C003E",  # Dark Purple
    "#003366",  # Dark Blue
    "#0A3D62",  # Dark Teal
    "#004225",  # Dark Green
    "#4B3F00",  # Dark Yellow
    "#660000",  # Dark Red
    "#401A00",  # Dark Orange
    "#2E1503",  # Dark Brown
    "#2D2D4B",  # Dark Indigo
    "#2C003E",  # Dark Magenta
    "#333333"   # Dark Neutral Gray
]
############################
# sample = sys.argv[1] 
sample = 'lung' 
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'+sample

############ read model results as adata 
picasa_adata = ad.read_h5ad(wdir+'/fig_0/results/picasa.h5ad')
wdir = wdir + '/fig_7'

####################################
import pandas as pd 
df = pd.read_csv(wdir+'/results/unique_space_selected_cells.csv.gz',compression='gzip',index_col=0)

from picasa.util.plots import get_colors
from plotnine import * 

df['cluster'] = ['u_'+str(x) for x in df['leiden']]

dfg = df.groupby(['cluster','batch']).count()['batch_id'].reset_index()
cluster_sum = dict(dfg.groupby('cluster')['batch_id'].sum())
dfg['ncount'] = [x/cluster_sum[y] for x,y in zip(dfg['batch_id'],dfg['cluster'])]


nlabel = dfg['cluster'].nunique()
custom_palette = dark_colors[:nlabel]
legend_size = 7

cluster_order = dfg['cluster'].unique()

dfg['cluster'] = pd.Categorical(dfg['cluster'], categories=cluster_order, ordered=True)

p = (ggplot(data=dfg, mapping=aes(x='cluster', fill='batch',weight='ncount')) +
geom_bar(position='stack') +
scale_color_manual(values=custom_palette)  +
labs(x="Cluster", y="Patient distribution") +
guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'),
    axis_text_x=element_text(rotation=45, hjust=1,size=20),
        axis_title_x=element_text(size=20),
        axis_title_y=element_text(size=20)
)

p.save(filename = wdir+'/results/patient_prop.pdf', height=10, width=15, units ='in', dpi=600)