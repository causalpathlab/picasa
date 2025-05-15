import os
import sys 

SAMPLE = sys.argv[1]
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/analysis/'


DATA_DIR = os.path.join(WDIR, SAMPLE, 'results')
RESULTS_DIR = os.path.join(WDIR, SAMPLE,'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)



import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

methods = ['umap_batch','umap_celltype','all']

fig, axes = plt.subplots(3, 1, figsize=(15, 8))
axes = axes.flatten()

for ax, label in zip(axes, methods):
    img_path = os.path.join(RESULTS_DIR, f'benchmark_plot_{label}.png')
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_frame_on(False)

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, "benchmark_plot_combine.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
