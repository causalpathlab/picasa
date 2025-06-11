
<div align="center">
    <img src="images/picasa_workflow.png" alt="Logo" width="500" height="500">
</div>



### This is a project repository for -
- Subedi, Sishir, and  Yongjin P. Park. "Decomposing patient heterogeneity of single-cell cancer data by cross-attention neural networks." [medRxiv 2025.06.04.25328900](https://www.medrxiv.org/content/10.1101/2025.06.04.25328900v1)


## Requirements

The following packages are required:

- anndata==0.10.8
- annoy==1.17.0
- numpy==1.24.4
- pandas>=2.0.3
- scanpy==1.9.3
- torch==2.5.1

We highly recommend to install `picasa` from PyPI in a new conda environment.

```
conda create --name picasa_env "python>=3.9"
conda activate picasa_env
pip install picasa
```

## Data

**Lung cancer**: The lung cancer dataset is available from [GSE148071](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148071). 

**Ovarian cancer**: The high-grade serous ovarian cancer (HGSOC) dataset is available from [GSE165897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165897). 

**Breast cancer**:The breast cancer single-cell dataset is available from [GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078). 

**Normal pancreas**: The normal pancreas dataset is available from `Seuret` data integration tutorial, [https://satijalab.org/seurat/archive/v3.2/integration.html](https://satijalab.org/seurat/archive/v3.2/integration.html).

**Simulation data**: The dataset is available from Figshare platform: [https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968).


## Tutorial

For the step-by-step tutorial, please refer to <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial">
notebooks </a>:

- Tutorial 1. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/1_training_picasa_model.ipynb">
 Training PICASA model using simulated datasets.</a>

- Tutorial 2. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/2_plotting_all_three_latent_umaps.ipynb">
Plotting all three latent representations learned by the model.</a>

- Tutorial 3. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/3_attention_matrix_analysis.ipynb">
Analysis of the cross attention matrix estimated by the model.</a>

- Tutorial 4. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/4_cancer_common_representation_analysis.ipynb">
Cancer common representation analysis.</a>

- Tutorial 5. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/5_cancer_unique_representation_analysis.ipynb">
Cancer unique representation analysis.</a>

- Tutorial 6. <a href="https://github.com/causalpathlab/picasa/tree/main/picasa_reproducibility/analysis/tutorial/6_cancer_patient_outcome_analysis.ipynb">
Cancer patient outcome analysis.</a>
