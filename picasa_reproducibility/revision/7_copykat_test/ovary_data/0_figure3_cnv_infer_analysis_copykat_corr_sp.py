import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sample = 'ovary'
pp = '/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/'
wdir = os.path.join(pp, sample)

def load_and_process_copykat(filename):
    filepath = os.path.join(filename)
    df = pd.read_csv(filepath, sep='\t')
    
    cols = [f"chr{x}_{y}" for x, y in zip(df['chrom'], df['chrompos'])]
    
    df_cells = df.iloc[:, 3:].T
    df_cells.columns = cols
    return df_cells

df_raw = load_and_process_copykat('raw_data_copykat_final_results_bin_by_cell.txt')
df_full = load_and_process_copykat('full_data_copykat_final_results_bin_by_cell.txt')
df_common = load_and_process_copykat('common_data_copykat_final_results_bin_by_cell.txt')
df_unique = load_and_process_copykat('unique_data_copykat_final_results_bin_by_cell.txt')

df_means = pd.concat({
    'Raw': df_raw.mean(),
    'Full': df_full.mean(),
    'Common': df_common.mean(),
    'Unique': df_unique.mean()
}, axis=1).dropna()

print(f"Number of perfectly aligned genomic bins across all 4 datasets: {len(df_means)}")


spearman_corr = df_means.corr(method='pearson')
print("--- Aligned spearman Correlation ---")
print(spearman_corr)

plt.figure(figsize=(6, 5))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f")
plt.title('Genomic Bin Correlation Across Conditions (Aligned)')
plt.tight_layout()
plt.savefig("picasa_cnv1.pdf")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_means, palette="Set2")
plt.title('Distribution of CNV Mean Values per Dataset')
plt.ylabel('Copy Number Signal')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("picasa_cnv2.pdf")


plt.figure(figsize=(16, 5))
plt.plot(df_means['Raw'].values, label='Raw Data', alpha=0.8, linewidth=1.5)
plt.plot(df_means['Full'].values, label='Full Data', alpha=0.8, linewidth=1.5)
plt.plot(df_means['Common'].values, label='Common Data', alpha=0.8, linewidth=1.5)
plt.plot(df_means['Unique'].values, label='Unique Data', alpha=0.8, linewidth=1.5)

plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.title('Mean CNV Signal Across Aligned Genomic Bins')
plt.xlabel('Genomic Bins Index')
plt.ylabel('Mean Copy Number Ratio / Signal')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("picasa_cnv3.pdf")
