import gseapy as gp
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for cluster in ['c_'+str(x) for x in range(10)]:
    
    print(cluster)
    
    df = pd.read_csv('results/sc_context_'+cluster+'.csv.gz',index_col=0)

    df.columns = ['f_'+str(x) for x in df.columns]

    ranked_gene_list = {}
    for factor in df.columns:
        ranked_df = df[factor].sort_values(ascending=False)
        ranked_gene_list[factor] = ranked_df.reset_index()  
        ranked_gene_list[factor].columns = ['Gene', 'Score']  


    gene_set_library = 'MSigDB_Hallmark_2020'  

    results = {}
    significant_gene_sets = {}
    pval_threshold = 0.01 

    for factor in df.columns:
        gsea_res = gp.prerank(rnk=ranked_gene_list[factor],  
                            gene_sets=gene_set_library,
                            min_size=15,  
                            max_size=500,  
                            permutation_num=1000,
                            outdir=None)  
        results[factor] = gsea_res
        
        significant_gene_sets[factor] = gsea_res.res2d[(gsea_res.res2d['FDR q-val'] < pval_threshold)]



    heatmap_data = []
    for factor, sig_sets in significant_gene_sets.items():
        sig_sets = sig_sets.reset_index()
        for pi,pathway in enumerate(sig_sets.Term):  
            heatmap_data.append([factor,pathway, sig_sets.loc[pi, 'NES']])

    df_hmap = pd.DataFrame(heatmap_data)
    df_hmap.columns = ['factor','pathway','nes']
    df_hmap = pd.pivot(df_hmap,columns='pathway',index='factor',values='nes')
    df_hmap.fillna(0, inplace=True)

    df_hmap.columns = [x[:20] for x in df_hmap.columns]
    # Plot the heatmap
    
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(20, 15))
    sns.heatmap(df_hmap.T, annot=False, cmap='vlag', cbar_kws={'label': 'NES (Normalized Enrichment Score)'})
    plt.title("Heatmap of Enrichment Scores (NES) for Factors and Pathways")
    plt.ylabel("Pathways")
    plt.xlabel("Factors")
    plt.xticks(rotation=90)
    plt.savefig('results/gsea'+cluster+'.png')

