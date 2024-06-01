import sys
sys.path.append("/home/BCCRC.CA/ssubedi/projects/experiments/picasa/")
import pandas as pd
import picasa.simulation.sim as sim


bulk_data = sys.argv[1]
sim_data_path = sys.argv[2]
rho = float(sys.argv[3])
depth = int(sys.argv[4])
size = int(sys.argv[5])
seed = int(sys.argv[6])


bulk_path = '/data/sishir/database/dice_immune_bulkrna/*.csv'

sim.simdata_from_bulk_copula(bulk_path,'data/'+sim_data_path,size,rho,depth,seed)
