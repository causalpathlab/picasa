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
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1] 
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'model_results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


from plotnine import *

def plot_lisi(df_lisi):
	
	df_summary = df_lisi.copy()
	df_summary["Method"] = pd.Categorical(df_summary["Method"], categories=['PCA','Combat','Harmony', 'Scanorama','Liger','scVI','DML','PICASA'], ordered=True)

	df_summary = pd.melt(df_summary,id_vars='Method')
	
	custom_colors = {
		"scVI": "#1f77b4",  # Blue
		"Combat": "#ff7f0e",     # Orange
		"PICASA": "#2ca02c",    # Green
		"Liger": "#d62728",      # Red
		"PCA": "#9467bd",        # Purple
		"Harmony": "#8c564b",     # Brown
		"Scanorama": "#e377c2",  # Pink
		"DML": "#17becf"         # Cyan
	}
	
	plot_pairs = ['NMI:ARI']
 
	for pair in plot_pairs:
		m1 = pair.split(':')[0]
		m2 = pair.split(':')[1]
  
		df_filtered = df_summary.loc[df_summary['variable'].isin([m1,m2]),:]
	
		p = (
		ggplot(df_filtered, aes(x="Method", y="value", group="variable", fill="Method",shape='variable'))
		+ geom_point(size=7.5) 
		+ geom_line(color='gray')
		+ scale_fill_manual(values=custom_colors)  
		+ labs(
			x="Method",
			y="Value",
			fill="Method",
			title="BioConservation - "+m1 +" and "+ m2
		)
		+ theme_minimal()  
		+ theme(
		figure_size=(10, 6),
		panel_background=element_rect(fill="white", color=None),  
		plot_background=element_rect(fill="white", color=None),
        axis_text=element_text(size=12, weight="bold",color='black'),
        axis_title=element_text(size=12, weight="bold"),
        legend_text=element_text(size=12, weight="bold"),
        legend_title=element_text(size=12, weight="bold"),
        plot_title=element_text(size=12, weight="bold", ha='center'),
        axis_text_x=element_text(size=12, weight="bold", angle=45, ha='right')  
		)
		)
		p.save(os.path.join(RESULTS_DIR,'benchmark_plot_'+pair+'.pdf'))
		plt.close()
		

	plot_pairs = [ 'ILISI_mean:ILISI_std']
 
	for pair in plot_pairs:
		m1, m2 = pair.split(':')  # Split into mean and std components

		# Filter and pivot data
		df_filtered = df_summary[df_summary['variable'].isin([m1, m2])]
		df_pivot = df_filtered.pivot(index="Method", columns="variable", values="value").reset_index()
		df_pivot['Method'] = df_pivot['Method'].astype('category')
  
		# Calculate ymin and ymax for error bars
		df_pivot['ymin'] = df_pivot[m1] - df_pivot[m2]
		df_pivot['ymax'] = df_pivot[m1] + df_pivot[m2]

		# Create the plot
		p = (
			ggplot(df_pivot, aes(x="Method", y=m1,color="Method"))
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
			+ labs(
				x="Method",
				title="BatchCorrection - "+ m1.split('_')[0]
			)
		+ theme(
		figure_size=(10, 6),
		panel_background=element_rect(fill="white", color=None),  
		plot_background=element_rect(fill="white", color=None),
        axis_text=element_text(size=12, weight="bold",color='black'),
        axis_title=element_text(size=12, weight="bold"),
        legend_text=element_text(size=12, weight="bold"),
        legend_title=element_text(size=12, weight="bold"),
        plot_title=element_text(size=12, weight="bold", ha='center'),
        axis_text_x=element_text(size=12, weight="bold", angle=45, ha='right')  
		)
		)
		print(pair)
		p.save(os.path.join(RESULTS_DIR,'benchmark_plot_'+pair+'.pdf'))
		plt.close()
  

def eval_plot():
	df_lisi = pd.read_csv(os.path.join(RESULTS_DIR,'benchmark_all_scores.csv'))
	df_lisi = df_lisi.drop(columns=['Unnamed: 0'])
	print(df_lisi.columns)
	df_lisi.rename(columns={'csil_res':'csil_score','method':'Method','ilisi_mean':'ILISI_mean','ilisi_std':'ILISI_std','nmi':'NMI','ari':'ARI'},inplace=True)
 
	mmap = {'pca': 'PCA',
		'combat': 'Combat',
		'harmony': 'Harmony',
		'scanorama': 'Scanorama',
		'liger': 'Liger',
		'scvi': 'scVI',
		'dml': 'DML',
		'picasa': 'PICASA'}
	df_lisi['Method'] = [mmap[x] for x in df_lisi['Method']]

	plot_lisi(df_lisi)



eval_plot()
