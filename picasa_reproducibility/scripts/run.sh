#!/bin/bash

# python 3_benchmark_ext_scanpy.py $1


# python 3_benchmark_ext_liger.py $1
# python 3_benchmark_ext_scvi.py $1

python 3_benchmark_picasa.py $1
python 4_bm_eval.py $1

python 5_eval_plot_box.py $1
python 5_eval_plot_umap.py $1
python 6_eval_plot_combine.py $1

