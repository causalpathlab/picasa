import os
import glob
import logging
import matplotlib.pyplot as plt
import seaborn as sn
import anndata as an
import pandas as pd
import scanpy as sc
import harmonypy as hm

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SAMPLE = sys.argv[1] 
# WDIR = sys.argv[2]

SAMPLE = 'sim6'
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'data')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'results')
os.makedirs(RESULTS_DIR, exist_ok=True)



import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

methods = ['combat', 'scvi', 'cellanova', 'harmony', 'pca', 'scanorama', 'liger', 'biolord']

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for ax, label in zip(axes, methods):
    img_path = os.path.join(RESULTS_DIR, f'scanpy_umap_{label}.png')
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(label)
    else:
        ax.set_title(f"{label}\n[Missing]")
        ax.axis('off')
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "scanpy_umap_combined.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined image saved to {output_path}")
