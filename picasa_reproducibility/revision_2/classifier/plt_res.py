import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('all_datasets_linear_evaluation_results.csv')
df.columns = df.columns.str.strip()


sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


dataset_map ={
  "brca" : "Breast (Tumour)",
  "lung" : "Lung (Tumour)",
  "pancreas" : "Pancreas (Normal)",
  "ovary" : "Ovary (Tumour)",
  "sim1" : "Sim1",
  "sim2" : "Sim2",
  "sim3" : "Sim3"
}
datasets = df['Dataset'].unique()
target_tasks = df['Target Task'].unique()
representations = df['Representation'].unique()

ncols = 2
num_datasets = len(datasets)
total_slots = num_datasets + (1 if num_datasets % ncols != 0 else 0)
nrows = (total_slots + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), sharey=True
)

axes = np.atleast_1d(axes).flatten()

palette = sns.color_palette('deep', len(representations))
rep_color = dict(zip(representations, palette))

for idx, dataset in enumerate(datasets):
  ax = axes[idx]
  sub_df = df[df['Dataset'] == dataset]

  x = np.arange(len(target_tasks))
  width = 0.22

  for i, rep in enumerate(representations):
    rep_data = sub_df[sub_df['Representation'] == rep]
    offset = (i - len(representations) / 2 + 0.5) * width

    means, stds = [], []
    for task in target_tasks:
      match = rep_data[rep_data['Target Task'] == task]
      if not match.empty:
        means.append(match['Mean Accuracy'].values[0])
        stds.append(match['Std Dev'].values[0])
      else:
        means.append(0)
        stds.append(0)

    ax.bar(
        x + offset,
        means,
        width,
        yerr=stds,
        label=rep,
        color=rep_color[rep],
        capsize=2.5,
        error_kw={'elinewidth': 0.9, 'ecolor': 'black', 'capthick': 0.9},
        edgecolor='black',
        linewidth=0.5,
    )

  ax.set_xticks(x)
  ax.set_xticklabels(target_tasks, fontweight='medium')
  ax.set_ylim(0, 1.05)
  ax.set_title(
      f'{dataset_map[dataset]}', fontsize=11, fontweight='bold', pad=8
  )
#   ax.set_xlabel('Target Task', fontsize=10, fontweight='medium')

  if idx % ncols == 0:
    ax.set_ylabel('Linear Probing Accuracy', fontsize=10, fontweight='medium')

legend_idx = num_datasets
if legend_idx < len(axes):
  legend_ax = axes[legend_idx]
  legend_ax.axis('off')  
  handles, labels = axes[0].get_legend_handles_labels()

  legend_ax.legend(
      handles,
      labels,
      title='Representation',
      loc='center',
      frameon=True,
      facecolor='white',
      edgecolor='none',
      fontsize=12,
      title_fontsize=11,
  )

for j in range(legend_idx + 1, len(axes)):
  fig.delaxes(axes[j])

# fig.suptitle(
#     'Evaluation of Latent Representation Decomposition',
#     fontsize=13,
#     weight='bold',
#     y=0.98,
# )

sns.despine(top=True, right=True)
plt.tight_layout()
fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.2)

plt.savefig('evaluation_results_grid.pdf', bbox_inches='tight', dpi=300)
plt.savefig('evaluation_results_grid.png', bbox_inches='tight', dpi=300)

plt.show()