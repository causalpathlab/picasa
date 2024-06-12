
rm data/sim3_*
snakemake -c 3 -s snakefile_data.smk 

mv data/sim_r*_s_1_*.h5 data/sim3_b1.h5
mv data/sim_r*_s_2_*.h5 data/sim3_b2.h5
mv data/sim_r*_s_3_*.h5 data/sim3_b3.h5

python 0_prep_data.py

