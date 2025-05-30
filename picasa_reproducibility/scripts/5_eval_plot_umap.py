import os
import logging
import matplotlib.pyplot as plt

import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/scripts/')

import constants 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


SAMPLE = sys.argv[1]
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)



import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

methods = ['pca','bbknn','combat','harmony', 'scanorama','liger','scvi','picasac','picasau','picasauc']

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for ax, label in zip(axes, methods):
    img_path = os.path.join(RESULTS_DIR, f'scanpy_{label}_umap_{constants.BATCH}.png')
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_frame_on(False)


plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "benchmark_plot_umap_"+constants.BATCH+".png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined image saved to {output_path}")


fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for ax, label in zip(axes, methods):
    img_path = os.path.join(RESULTS_DIR, f'scanpy_{label}_umap_{constants.GROUP}.png')
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_frame_on(False)
    else:
        ax.set_title(f"{label}\n[Missing]")
        ax.axis('off')
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "benchmark_plot_umap_"+constants.GROUP+".png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
