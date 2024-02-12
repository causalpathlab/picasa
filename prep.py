import anndata as ad
import picasa

rna = ad.read_h5ad('data/brca_scrna.h5ad')
spatial = ad.read_h5ad('data/brca_spatial.h5ad')


pico = picasa.create_picasa_object({'rna':rna,'spatial':spatial})


pico.update_common_features()