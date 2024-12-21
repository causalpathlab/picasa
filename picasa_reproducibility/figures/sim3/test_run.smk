import shutil
onsuccess:
    shutil.rmtree(".snakemake")

sample = 'sim3'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/picasa/picasa_reproducibility/figures/'


ED = [2000,3000]
AD = [15,30]
PL = [30,60]
LR = [1e-3,1e-4,1e-5,1e-6,1e-7]
PW = [0.0,0.01,0.1,0.2,0.4,0.6,0.8,1.0]



sim_data_pattern = '_ed_{ed}_ad_{ad}_pl_{pl}_lr_{lr}_pw_{pw}_'

pdir = wdir+sample+'/'+sim_data_pattern+'/'
pdir_r1 = wdir+sample+'/'+sim_data_pattern+'/'+sample
pdir_r2 = wdir+sample+'/'+sim_data_pattern+'/'+sample+'/results/'


rule all:
    input:
        expand(pdir_r2+sample+sim_data_pattern+'.png',ed=ED,ad=AD,pl=PL,lr=LR,pw=PW)

rule run_nmf:
    input:
        script = '1_picasa_run_test.py'
    output:
        umap = pdir_r2+sample+sim_data_pattern+'.png'
    params:
        sample = sample,
        wdir = wdir
    shell:
        'python  {input.script}  {params.sample} {params.wdir} {wildcards.ed} {wildcards.ad} {wildcards.pl} {wildcards.lr} {wildcards.pw}'