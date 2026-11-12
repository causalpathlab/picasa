import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

samples = ["brca", "lung", "ovary","pancreas","sim1","sim2","sim3"]


def get_accuracy_with_std(X, y_labels, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    scores = cross_val_score(clf, X, y_labels, cv=cv, scoring='accuracy')
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    return mean_acc, std_acc

all_records = []

for sample in samples:
    file_path = f"/Users/sishirsubedi/Documents/projects/picasa/picasa_reproducibility/analysis/{sample}/results/picasa.h5ad"
    print(f"\n==============================")
    print(f" Processing Dataset: {sample}")
    print(f"==============================")
    
    try:
        # Load data
        adata = sc.read_h5ad(file_path)

        # test subsample to 5,000 cells for fast execution
        # if adata.n_obs > 5000:
        #     sc.pp.subsample(adata, n_obs=5000, random_state=42)

        # just the top 10 most frequent cell types for balance
        top_cell_types = adata.obs["celltype"].value_counts().nlargest(10).index
        adata = adata[adata.obs["celltype"].isin(top_cell_types)].copy()

        cell_type = adata.obs["celltype"].astype(str).to_numpy()
        patient_label = adata.obs["batch"].astype(str).to_numpy()

        z = adata.obsm["common"]
        y = adata.obsm["unique"]
        combined = np.hstack([z, y])

        tasks = [("Cell Type", cell_type), ("Patient / Batch", patient_label)]
        representations = [
            ("Z (common)", z),
            ("Y (unique)", y),
            ("Combined (Z + Y)", combined),
        ]

        for task_name, labels in tasks:
            for rep_name, X in representations:
                mean_acc, std_acc = get_accuracy_with_std(X, labels)
                acc_str = f"{mean_acc:.3f} ± {std_acc:.3f}"
                
                all_records.append({
                    "Dataset": sample,
                    "Target Task": task_name,
                    "Representation": rep_name,
                    "Mean Accuracy": round(mean_acc, 3),
                    "Std Dev": round(std_acc, 3),
                    "Accuracy (Formatted)": acc_str
                })
                print(f"  [{task_name}] {rep_name}: {acc_str}")
                
    except Exception as e:
        print(f"Error processing dataset '{sample}': {e}")

df_master = pd.DataFrame(all_records)
output_filename = "all_datasets_linear_evaluation_results.csv"
df_master.to_csv(output_filename, index=False)

print(f"successfully saved to '{output_filename}' in your current working directory.")