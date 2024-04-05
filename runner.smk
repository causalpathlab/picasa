
import shutil
onsuccess:
    shutil.rmtree(".snakemake")


rule all:
    input:
        expand('data/sim/sim_sp.h5ad'),
        expand('data/sim/sim_sc.h5ad')

rule sc_simulated_data:
    input:
        script = '1_simulate_data.py',
        in_sp = 'data/sim/brcasim_sp.h5ad',
        in_sc = 'data/sim/brcasim_sc.h5ad'
    output:
        out_sp = 'data/sim/sim_sp.h5ad',
        out_sc = 'data/sim/sim_sc.h5ad'
    shell:
        'python {input.script}'
