import shutil
onsuccess:
    shutil.rmtree(".snakemake")

# configfile: 'config.yaml'

sample = 'sim'
input_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/znode/sim3/data/'
output_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/znode/sim3/results/'
scripts_dir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/znode/sim3/'

RHO = [0.5] 
DEPTH = [10000]
SIZE = [500] ## total cell ~3k
SEED = [1,2,3]
TOPIC = [13]
RES =[0.5]

sim_data_pattern = '_r_{rho}_d_{depth}_s_{size}_s_{seed}_t_{topic}_r_{res}'
sim_data_pattern = sample + sim_data_pattern


rule all:
    input:
        expand(input_dir + sim_data_pattern+'.h5',rho=RHO,depth=DEPTH,size=SIZE,seed=SEED,topic=TOPIC,res=RES)

rule sc_simulated_data:
    input:
        script = scripts_dir + 'step1_data.py',
        bulk_data = '/data/sishir/database/dice_immune_bulkrna/CD8_NAIVE_TPM.csv'
    output:
        sim_data = input_dir + sim_data_pattern+'.h5'
    params:
        sim_data_pattern = sim_data_pattern
    shell:
        'python {input.script} {input.bulk_data} {params.sim_data_pattern} {wildcards.rho} {wildcards.depth} {wildcards.size} {wildcards.seed}'

