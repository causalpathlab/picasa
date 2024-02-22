
import matplotlib.pylab as plt
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import picasa


sc_ref_path = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/fig1/data/sc.h5ad'
sp_ref_path = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/figures/fig1/data/sp.h5ad'

dfsc,dfsp,nbrs = picasa.sim.generate_simdata(sc_ref_path,sp_ref_path)

dfsp.to_csv('dataspmap.csv')