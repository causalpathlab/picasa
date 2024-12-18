
import shutil
onsuccess:
    shutil.rmtree(".snakemake")

SAMPLE='sim2'

WDIR='/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

OUTDIR=WDIR+SAMPLE+'/results/'

rule all:
    input:
        expand(OUTDIR+'picasa_common_train_loss.txt.gz'),
        expand(OUTDIR+'picasa_common_train_loss.png'),
        expand(OUTDIR+'picasa_common.model'),
        expand(OUTDIR+'picasa_unique_train_loss.txt.gz'),
        expand(OUTDIR+'picasa_unique_train_loss.png'),
        expand(OUTDIR+'picasa_unique.model'),
        expand(OUTDIR+'picasa_base_train_loss.txt.gz'),
        expand(OUTDIR+'picasa_base_train_loss.png'),
        expand(OUTDIR+'picasa_base.model'),
        expand(OUTDIR+'picasa.h5ad'),
        expand(OUTDIR+'picasa_common_umap.png'),
        expand(OUTDIR+'picasa_unique_umap.png'),
        expand(OUTDIR+'picasa_base_umap.png'),
        expand(OUTDIR+'benchmark_pca.csv.gz'),
        expand(OUTDIR+'benchmark_combat.csv.gz'),
        expand(OUTDIR+'benchmark_scanorama.csv.gz'),
        expand(OUTDIR+'benchmark_harmony.csv.gz'),
        expand(OUTDIR+'benchmark_scvi.csv.gz'),
        expand(OUTDIR+'benchmark_cellanova.csv.gz'),
        expand(OUTDIR+'benchmark_liger.csv.gz'),
        expand(OUTDIR+'benchmark_biolord.csv.gz'),
        expand(OUTDIR+'benchmark_plot_lisi_group.png'),
        expand(OUTDIR+'benchmark_plot_umap_batch.png')

rule train_model:
    input:
        script = WDIR+SAMPLE+'/1_picasa_run.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        tc = OUTDIR+'picasa_common_train_loss.txt.gz',
        tc_i = OUTDIR+'picasa_common_train_loss.png',
        tc_m = OUTDIR+'picasa_common.model',
        tu = OUTDIR+'picasa_unique_train_loss.txt.gz',
        tu_i = OUTDIR+'picasa_unique_train_loss.png',
        tu_m = OUTDIR+'picasa_unique.model',
        tb = OUTDIR+'picasa_base_train_loss.txt.gz',
        tb_i = OUTDIR+'picasa_base_train_loss.png',
        tb_m = OUTDIR+'picasa_base.model',
        t_d = OUTDIR+'picasa.h5ad'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule model_analysis:
    input:
        script = '2_picasa_analysis.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        c_umap = OUTDIR+'picasa_common_umap.png',
        u_umap = OUTDIR+'picasa_unique_umap.png',
        b_umap = OUTDIR+'picasa_base_umap.png'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule scanpy_external_analysis:
    input:
        script = '3_1_benchmark_ext_scanpy.py'
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        outc = OUTDIR+'benchmark_combat.csv.gz',
        outh = OUTDIR+'benchmark_harmony.csv.gz',
        outs = OUTDIR+'benchmark_scanorama.csv.gz',
        outp = OUTDIR+'benchmark_pca.csv.gz'
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
        out = OUTDIR+'benchmark_scvi.csv.gz'
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
        out = OUTDIR+'benchmark_cellanova.csv.gz'
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
        out = OUTDIR+'benchmark_liger.csv.gz'
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
        out = OUTDIR+'benchmark_biolord.csv.gz'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule eval:
    input:
        script = '4_eval.py',
        picasa = rules.train_model.output.t_d,
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
        out_lisi = OUTDIR+'benchmark_plot_lisi_group.png'
        # out_clust = OUTDIR+'benchmark_lisi.png'
        # out_asw = OUTDIR+'benchmark_lisi.png'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

rule eval_viz:
    input:
        script = '4_eval_viz_umap.py',
        eval_out = rules.eval.output.out_lisi
    params:
        sample = SAMPLE,
        wdir = WDIR
    output:
        out_i = OUTDIR+'benchmark_plot_umap_batch.png'
    shell:
        'python {input.script} {params.sample} {params.wdir}'

