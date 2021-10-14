# LinTS Reproducibility on Simulated Data

This directory contains the notebooks required to create the data and metrics for a synthetic dataset, in a similar manner to tables 2 and 3.
Tables 5 and 6 below show how the reproducibility problem can be created using simulated recommendation data, and how using Cholesky decomposition creates systematic and reproducible results.
This folder replicates these findings using a synthetic dataset.

## Notebooks
You can run the notebooks linked below in the environments they are named after to generate deterministic numbers.

- Run the [Linux RedHat OpenBLAS](LinuxRedHat_OpenBLAS.ipynb) notebook when using the [Linux RedHat OpenBLAS](../../environments/LinuxRedHat_OpenBLAS) environment.
- Run the [Linux Sagemaker LAPACK](LinuxSagemaker_LAPACK.ipynb) notebook when using the [Linux Sagemaker LAPACK](../../environments/LinuxSagemaker_LAPACK) environment.
- Run the [MacOS Darwin MKL](MacOSDarwin_MKL.ipynb) notebook when using the [MacOS Darwin MKL](../../environments/MacOSDarwin_MKL) environment.
- Run the [MacOS Darwin OpenBLAS](MacOSDarwin_OpenBLAS.ipynb) notebook when using the [MacOS Darwin OpenBLAS](../../environments/MacOSDarwin_OpenBLAS) environment.
- Run the [Windows OpenBLAS](Windows_OpenBLAS.ipynb) notebook when using the [Windows OpenBLAS](../../environments/Windows_OpenBLAS) environment.

Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment.
This notebook will generate the numbers for a simulated dataset, in the appropriately marked locations of the notebook.

## Data
- `simulated_data.csv` contains the simulated data with 4 arms

We simulate a scenario with a 4-armed bandit that recommends a single item at each decision for the given context.
To create this data set, we utilize the `Scikit-learn` `make_classification` function to generate binary classification observations.
This function returns the context features and class (0/1).
To create separate contexts that can predict each arm, we generate 100 binary classification observations 4 times, using the arm index as the seed to obtain unique contexts.
We confirm that ![equation](http://latex.codecogs.com/png.latex?%5Cinline%20A%5E%7B-1%7D) for each arm contains duplicate singular values.
The train-test split follows the 70-30 data regime.

## Results

### Table 5
| Environment | Comparison |     |     |     |     |
| ----------- | ---------- | --- | --- | --- | --- |
| Red Hat Linux | Score<br>Probability<br>Prediction | 27%<br>34%<br>99% | | | |
| MacOS MKL | Score<br>Probability<br>Prediction | 13%<br>14%<br>91% | 13%<br>14%<br>92% | | |
| MacOS OpenBLAS | Score<br>Probability<br>Prediction | 27%<br>34%<br>99% | **100%**<br>**100%**<br>**100%** | 13%<br>14%<br>92% | |
| Amazon AWS Linux | Score<br>Probability<br>Prediction | 45%<br>50%<br>99% | 27%<br>35%<br>**100%** | 13%<br>14%<br>92% | 27%<br>35%<br>**100%** |
| | | **Windows** | **Red Hat Linux** | **Mac MKL** | **Mac OpenBLAS** | 

Numerical Results on Synthetic Data:
Comparison of the number of matching scores, probabilities, and predictions across all pairs of environments when SVD-based LinTS implementation is used.
A reproducible result should match 480 times (100%) as shown in **bold**.
Any other result is a discrepancy due to non-deterministic behavior.

The scores, probabilities, and predictions for Cholesky match **100%** for all environment pairs.

### Table 6
| Environment | Clicks | Accuracy | Precision | Recall | AUC | CTR@1 | NDCG@1 | NDCG@4 |
| ----------- | ------ | -------- | --------- | ------ | --- | ----- | ------ | ------ |
| Windows | 261 | 0.650 | 0.776 | 0.608 | 0.663 | 0.389 | 0.121 | 0.620 |
| Red Hat Linux | 259 | 0.650 | 0.776 | 0.608 | 0.663 | 0.421 | 0.138 | 0.630 |
| MacOS MKL | 256 | 0.650 | 0.741 | 0.614 | 0.657 | 0.476 | 0.172 | 0.632 |
| MacOS OpenBLAS | 259 | 0.650 | 0.776 | 0.608 | 0.663 | 0.421 | 0.138 | 0.630 |
| Amazon AWS Linux | 259 | 0.650 | 0.776 | 0.608 | 0.663 | 0.421 | 0.138 | 0.630 |
| Cholesky (for all) | 230 | 0.425 | 0.482 | 0.417 | 0.425 | 0.388 | 0.241 | 0.632 |

Numerical Results on Synthetic Data:
Comparison of recommender system evaluation metrics across different environments when SVD-based LinTS implementation is used.
The last row shows the correct values from the deterministic Cholesky-based implementation that is identical for all environments.
