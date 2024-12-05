
import shutil
onsuccess:
    shutil.rmtree(".snakemake")


rule all:
    input:
        expand('results/picasa_common_train_loss.txt.gz'),
        expand('results/picasa_common_train_loss.png'),
        expand('results/picasa_common.model'),

        expand('results/picasa_unique_train_loss.txt.gz'),
        expand('results/picasa_unique_train_loss.png'),
        expand('results/picasa_unique.model'),

        expand('results/picasa_base_train_loss.txt.gz'),
        expand('results/picasa_base_train_loss.png'),
        expand('results/picasa_base.model'),

        expand('results/picasa.h5ad')

rule train_model:
    input:
        script = '1_picasa_run.py',
    output:
        
        tc = 'results/picasa_common_train_loss.txt.gz',
        tc_i = 'results/picasa_common_train_loss.png',
        tc_m = 'results/picasa_common.model',

        tu = 'results/picasa_unique_train_loss.txt.gz',
        tu_i = 'results/picasa_unique_train_loss.png',
        tc_m = 'results/picasa_unique.model',

        tb = 'results/picasa_base_train_loss.txt.gz',
        tb_i = 'results/picasa_base_train_loss.png',
        tc_m = 'results/picasa_base.model',

        t_d = 'results/picasa.h5ad'

    shell:
        'python {input.script}'

rule model_analysis:
    input:
        script = '2_picasa_analysis.py',
    output:
        
        c_umap = 'results/picasa_common_umap.png',
        u_umap = 'results/picasa_unique_umap.png',
        b_umap = 'results/picasa_base_umap.png'

    shell:
        'python {input.script}'
