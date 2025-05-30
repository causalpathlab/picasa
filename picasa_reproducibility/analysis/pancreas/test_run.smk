import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'pancreas'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'


ED = [3000]
AD = [15]
PL = [25]
LR = [1e-3]
# PW = [0.5,0.6,0.7,0.8,0.9,1.0]
PW = [1.1,1.2,1.5,2.0]


sim_data_pattern = '_ed_{ed}_ad_{ad}_pl_{pl}_lr_{lr}_pw_{pw}_'

pdir = wdir+sample+'/'+sim_data_pattern+'/'
pdir_r1 = wdir+sample+'/'+sim_data_pattern+'/'+sample
pdir_r2 = wdir+sample+'/'+sim_data_pattern+'/'+sample+'/results/'


rule all:
    input:
        expand(pdir_r2+sample+sim_data_pattern+'.png',ed=ED,ad=AD,pl=PL,lr=LR,pw=PW)

rule run_picasa:
    input:
        script = '1_picasa_run_test.py'
    output:
        umap = pdir_r2+sample+sim_data_pattern+'.png'
    params:
        sample = sample,
        wdir = wdir
    shell:
        'python  {input.script}  {params.sample} {params.wdir} {wildcards.ed} {wildcards.ad} {wildcards.pl} {wildcards.lr} {wildcards.pw}'