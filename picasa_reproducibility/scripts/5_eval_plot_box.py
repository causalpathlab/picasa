import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa')



SAMPLE = sys.argv[1] 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


from plotnine import *

def plot_lisi(df_res):
    
    df_summary = df_res.copy()
    df_summary["Method"] = pd.Categorical(df_summary["Method"], categories=['PCA','BBKNN','Combat','Harmony', 'Scanorama','Liger','scVI','PICASA_C','PICASA_U','PICASA_UC'], ordered=True)

    df_summary = pd.melt(df_summary,id_vars='Method')
    
    custom_colors = {
        "scVI": "#1f77b4",  # Blue
        "Combat": "#ff7f0e",     # Orange
        "PICASA_B": "palegreen",    # Green
        "PICASA_U": "green",    # Green
        "PICASA_UC": "yellowgreen",    # Green
        "PICASA_C": "limegreen",    # Green
        "Liger": "#d62728",      # Red
        "PCA": "#9467bd",        # Purple
        "BBKNN":"skyblue",
        "Harmony": "#8c564b",     # Brown
        "Scanorama": "#e377c2",  # Pink
    }
    

    plot_pairs = [ 'ILISI_mean:ILISI_std','CLISI_mean:CLISI_std']
    
    for pair in plot_pairs:
        
        m1, m2 = pair.split(':') 
        if 'I' in m1.split('_')[0][0]: tn = 'Batch correction'
        elif 'C' in m1.split('_')[0][0]: tn = 'Cell type'

        # Filter and pivot data
        df_filtered = df_summary[df_summary['variable'].isin([m1, m2])]
        df_pivot = df_filtered.pivot(index="Method", columns="variable", values="value").reset_index()
        df_pivot['Method'] = df_pivot['Method'].astype('category')
  
        # Calculate ymin and ymax for error bars
        df_pivot['ymin'] = df_pivot[m1] - df_pivot[m2]
        df_pivot['ymax'] = df_pivot[m1] + df_pivot[m2]

        print(df_pivot)
        # Create the plot
        p = (
            ggplot(df_pivot, aes(x="Method", y=m1,color="Method",ymin=0.0))
            + geom_point(aes(fill="Method"), size=7.5, color="black", stroke=0.5)
            + geom_errorbar(
                aes(ymin="ymin", ymax="ymax"),
                width=0.25,
                color="black",
                size=1.0       
            )
            + scale_color_manual(values=custom_colors)
            + scale_fill_manual(values=custom_colors)  
            + theme_minimal()
            + labs(x="",y="",title= tn+" LISI")
        + scale_y_continuous(expand=(0,0))
        + theme(
        figure_size=(10, 6),
        panel_background=element_rect(fill="white", color=None),  
        plot_background=element_rect(fill="white", color=None),
        axis_text=element_text(color='black'),
        axis_title=element_text(size=20, weight="bold"),
        legend_text=element_text(size=20, weight="bold"),
        legend_title=element_text(size=20),
        plot_title=element_text(size=20, weight="bold", ha='center'),
        axis_text_x=element_text(size=20, angle=45, ha='right'),  
        axis_text_y=element_text(size=20)  
        )
        )
        print(pair)
        
        p.save(os.path.join(RESULTS_DIR,'benchmark_plot_'+pair.replace(':','_')+'.pdf'))
        plt.close()
  

def eval_plot(df_res):
    
    df_res.rename(columns={'ilisi_mean':'ILISI_mean','ilisi_std':'ILISI_std','clisi_mean':'CLISI_mean','clisi_std':'CLISI_std'},inplace=True)
    
    mmap = {'pca': 'PCA',
        'bbknn':'BBKNN',
        'combat': 'Combat',
        'harmony': 'Harmony',
        'scanorama': 'Scanorama',
        'liger': 'Liger',
        'scvi': 'scVI',
        'picasab': 'PICASA_B',
        'picasac': 'PICASA_C',
        'picasau': 'PICASA_U',  
        'picasauc': 'PICASA_UC'  
  }
    df_res['Method'] = [mmap[x] for x in df_res['Method']]
    df_res = df_res[df_res['Method']!='PICASA_B']
    plot_lisi(df_res)



PATTERN = f'benchmark_*_scores.csv'
method_files = glob.glob(os.path.join(RESULTS_DIR, PATTERN))

df_res = pd.DataFrame()
for mf in method_files:
    dfc = pd.read_csv(mf)
    df_res = pd.concat([df_res,dfc])


eval_plot(df_res)

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

methods = ['ILISI_mean_ILISI_std','CLISI_mean_CLISI_std']

fig, axes = plt.subplots(1, 2)
axes = axes.flatten()

for ax, label in zip(axes, methods):
    img_path = os.path.join(RESULTS_DIR, f'benchmark_plot_{label}.png')
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        # ax.set_title(label)
    else:
        # ax.set_title(f"{label}\n[Missing]")
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    ax.axis('off')
for ax in axes[len(methods):]:
    ax.axis('off')  # Ensure the remaining axes are blank

plt.tight_layout()
# output_path = os.path.join(RESULTS_DIR, "benchmark_plot_all_two.pdf")
output_path = os.path.join(RESULTS_DIR, "benchmark_plot_all.pdf")
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.close()