import sys 
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/picasa')

import matplotlib.pylab as plt
import seaborn as sns
import anndata as an
import pandas as pd
import numpy as np
import picasa
import torch
import logging


import glob
import os



def plot_survival(df,fpath,score_col='score'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from sklearn.preprocessing import MinMaxScaler


    threshold = df[score_col].median()  

    df = df.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    df['OS_STATUS'] = [int(x.split(':')[0]) for x in df['OS_STATUS']]
    df['OS_STATUS'] = df['OS_STATUS'].astype(int)

    df['score_group'] = np.where(df[score_col] > threshold, 'High Score', 'Low Score')

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))

    for group in df['score_group'].unique():
        group_data = df[df['score_group'] == group]
        kmf.fit(durations=group_data['OS_MONTHS'], 
                event_observed=group_data['OS_STATUS'], 
                label=group)
        kmf.plot_survival_function(ci_show=True)


    high_score_data = df[df['score_group'] == 'High Score']
    low_score_data = df[df['score_group'] == 'Low Score']

    # Perform log-rank test
    results = logrank_test(
        high_score_data['OS_MONTHS'], low_score_data['OS_MONTHS'], 
        event_observed_A=high_score_data['OS_STATUS'], 
        event_observed_B=low_score_data['OS_STATUS']
    )

    # Get p-value
    p_value = results.p_value
    print(f"Log-Rank Test p-value: {p_value}")

    # Add p-value to the plot
    plt.title('Kaplan-Meier Survival Curve by Score Groups')
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(title='Score Group')

    # Display the p-value on the plot
    plt.text(
        x=plt.xlim()[1] * 0.5,  # Place text at 50% of the x-axis range
        y=0.8,                  # Slightly above 0 on the y-axis
        s=f"Log-Rank Test p-value: {p_value:.4f}",
        fontsize=12,
        color="red"
    )


    plt.title('Kaplan-Meier Survival Curve by Score Groups')
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(title='Score Group')
    plt.savefig(fpath)
    plt.close()
 
def get_ensembl_ids(gene_list):

    import mygene

    mg = mygene.MyGeneInfo()
    result = mg.querymany(gene_list, scopes="symbol", fields="ensembl.gene", species="human")

    gene_to_ensembl = {}
    for entry in result:
        query = entry['query']
        if 'ensembl' in entry:
            if isinstance(entry['ensembl'], list):
                ensembl_ids = [ens['gene'] for ens in entry['ensembl']]
            else:
                ensembl_ids = [entry['ensembl']['gene']]
        else:
            ensembl_ids = None
        gene_to_ensembl[query] = ensembl_ids

    eids = []
    for e in gene_to_ensembl.keys():
        if gene_to_ensembl[e] is not None:
            eids.append(gene_to_ensembl[e][0])
        else:
            eids.append(e)
    return eids

def get_hugo_gene_name(ensembl_ids):

    import mygene

    mg = mygene.MyGeneInfo()
    
    result = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human")


    ensembl_to_hugo = {}
    for entry in result:
        query = entry['query']
        if 'symbol' in entry:
            if isinstance(entry['symbol'], list):
                hugo_names = entry['symbol']
            else:
                hugo_names = [entry['symbol']]
        else:
            hugo_names = None
        ensembl_to_hugo[query] = hugo_names

    final_hugo_mapping = {}
    for ensembl_id, hugo_names in ensembl_to_hugo.items():
        if hugo_names is not None:
            final_hugo_mapping[ensembl_id] = hugo_names  
        else:
            final_hugo_mapping[ensembl_id] = None


    gnames = []
    for e in final_hugo_mapping.keys():
        if final_hugo_mapping[e] is not None:
            gnames.append(final_hugo_mapping[e][0])
        else:
            gnames.append(e)
    return gnames




####### get model params #####################
sample = 'ovary'
wdir = 'znode/ovary/'

directory = wdir+'/data'
pattern = 'ovary_*.h5ad'

file_paths = glob.glob(os.path.join(directory, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace('ovary_','')] = an.read_h5ad(wdir+'data/'+file_name)
	batch_count += 1
	if batch_count >=12:
		break


file_name = file_names[0].replace('.h5ad','').replace('ovary_','')

picasa_object = picasa.pic.create_picasa_object(
	batch_map,'unq',
	wdir)

b_sample = list(picasa_object.data.adata_list.keys())[0]
input_dim = picasa_object.data.adata_list[b_sample].X.shape[1]
enc_layers = [128,15]
unique_latent_dim = 15
common_latent_dim = 15
dec_layers = [128,128]
device='cpu'
num_batches = len(picasa_object.adata_keys)
picasa_unq_model = picasa.model.nn_unq.PICASAUNET(input_dim,common_latent_dim,unique_latent_dim,enc_layers,dec_layers,num_batches).to(device)

picasa_unq_model.load_state_dict(torch.load(wdir+'results/nn_unq.model', map_location=torch.device(device)))

picasa_unq_model.eval()

state_dict = picasa_unq_model.state_dict() 
parameter_name = "zinb_dispersion"  

if parameter_name in state_dict:
    param_tensor = state_dict[parameter_name].cpu().numpy()  
else:
    raise KeyError(f"Parameter '{parameter_name}' not found in the model state_dict.")

df_disp = pd.DataFrame(param_tensor)
df_disp.index = picasa_object.data.adata_list[b_sample].var.index.values

df_disp = np.exp(df_disp)


####### get patient data #####################

df_patient = pd.read_csv('../temp/tcga_ov_expr.csv.gz')
df_patient.set_index('Unnamed: 0',inplace=True)



####### align genes #####################

patient_genes = [ x.split('_')[1] for x in df_patient.columns]
model_genes = [x for x in df_disp.index.values]


match_genes_index = [ True if x in model_genes else False for x in patient_genes]
np.array(match_genes_index).sum()

match_genes = np.array(patient_genes)[match_genes_index]


df_patient = df_patient.iloc[:,match_genes_index]
df_disp = df_disp.loc[match_genes,:]

df_disp.shape
df_patient.shape


################ survival analysis #############


# normalized_df = df_patient.div(df_patient.sum(axis=0), axis=1) * 10000
# normalized_df.fillna(0.0,inplace=True)
# df_patient = np.log1p(normalized_df)

score = np.dot(df_patient,df_disp)

df_score = pd.DataFrame(score,index=df_patient.index)

df_score.reset_index(inplace=True)
df_score.columns = ['patient','score']
df_score.to_csv(wdir+'results/df_score.txt.gz',compression='gzip')


df_pmeta = pd.read_csv('../temp/data_clinical_patient.txt',sep='\t',skiprows=4,index_col=0)

df_score['PATIENT_ID'] =[x.split('-')[0]+'-'+x.split('-')[1]+'-'+x.split('-')[2] for x in df_score['patient']]

df_merge = pd.merge(df_score,df_pmeta,left_on='PATIENT_ID',right_index=True,how='left')

   
plot_survival(df_merge,fpath = wdir+'results/survival.png')