
import shutil
onsuccess:
    shutil.rmtree(".snakemake")

SAMPLE='sim2'

WDIR='/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'

OUTDIR=WDIR+SAMPLE+'/results/'

rule all:
    input:
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
        expand(OUTDIR+'benchmark_all_scores.csv'),
        expand(OUTDIR+'benchmark_plot_umap_batch.png')


rule picasa_analysis:
    input:
        script = '2_picasa_analysis.py',
        infile = OUTDIR+'picasa.h5ad'
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
        picasa = rules.picasa_analysis.output.c_umap,
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
        out_lisi = OUTDIR+'benchmark_all_scores.csv'
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

