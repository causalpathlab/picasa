import anndata as ad
import picasa

atac = ad.read_h5ad('data/brca_atac.h5ad')
rna = ad.read_h5ad('data/brca_scrna.h5ad')
spatial = ad.read_h5ad('data/brca_spatial.h5ad')


pico = picasa.create_picasa_object({'atac':atac,'rna':rna,'spatial':spatial})

gtf = '/data/sishir/database/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz'
gtf_by = 'gene_name'
pico.annotate_features(gtf,gtf_by)

pico.update_common_features()