
import shutil
onsuccess:
    shutil.rmtree(".snakemake")

SAMPLE = 'sim6'
WDIR = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'


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
        expand('results/picasa.h5ad'),
        expand('results/benchmark_pca.csv.gz'),
        expand('results/benchmark_combat.csv.gz'),
        expand('results/benchmark_scanorama.csv.gz'),
        expand('results/benchmark_harmony.csv.gz'),
        expand('results/benchmark_scvi.csv.gz'),
        expand('results/benchmark_cellanova.csv.gz'),
        expand('results/benchmark_liger.csv.gz'),
        expand('results/benchmark_biolord.csv.gz'),
        expand('results/benchmark_lisi.png')

rule train_model:
    input:
        script = '1_picasa_run.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        tc = 'results/picasa_common_train_loss.txt.gz',
        tc_i = 'results/picasa_common_train_loss.png',
        tc_m = 'results/picasa_common.model',
        tu = 'results/picasa_unique_train_loss.txt.gz',
        tu_i = 'results/picasa_unique_train_loss.png',
        tu_m = 'results/picasa_unique.model',
        tb = 'results/picasa_base_train_loss.txt.gz',
        tb_i = 'results/picasa_base_train_loss.png',
        tb_m = 'results/picasa_base.model',
        t_d = 'results/picasa.h5ad'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule model_analysis:
    input:
        script = '2_picasa_analysis.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        c_umap = 'results/picasa_common_umap.png',
        u_umap = 'results/picasa_unique_umap.png',
        b_umap = 'results/picasa_base_umap.png'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule scanpy_external_analysis:
    input:
        script = '3_1_benchmark_ext_scanpy.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        outc = 'results/benchmark_combat.csv.gz',
        outh = 'results/benchmark_harmony.csv.gz',
        outs = 'results/benchmark_scanorama.csv.gz',
        outp = 'results/benchmark_pca.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule scvi_analysis:
    conda:
        'scvi-env'
    input:
        script = '3_2_benchmark_ext_scvi.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out = 'results/benchmark_scvi.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule cellanova_analysis:
    conda:
        'scvi-env'
    input:
        script = '3_2_benchmark_ext_cellanova.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out = 'results/benchmark_cellanova.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule liger_analysis:
    conda:
        'scvi-env'
    input:
        script = '3_2_benchmark_ext_liger.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out = 'results/benchmark_liger.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule biolord_analysis:
    conda:
        'scvi-env'
    input:
        script = '3_2_benchmark_ext_biolord.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out = 'results/benchmark_biolord.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule eval:
    input:
        script = '4_eval.py',
        picasa = rules.train_model.output.t_d
        combat = rules.scanpy_external_analysis.output.outc,
        harmony = rules.scanpy_external_analysis.output.outh,
        scanorama = rules.scanpy_external_analysis.output.outs,
        pca = rules.scanpy_external_analysis.output.outp,
        scvi = rules.scvi_analysis.output.out,
        cellanova = rules.cellanova_analysis.output.out,
        liger = rules.liger_analysis.output.out,
        biolord = rules.biolord_analysis.output.out
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out_lisi = 'results/benchmark_lisi.png'
        # out_clust = 'results/benchmark_lisi.png'
        # out_asw = 'results/benchmark_lisi.png'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

