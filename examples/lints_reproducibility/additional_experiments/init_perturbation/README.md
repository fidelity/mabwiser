## Experiments with perturbation at model initialization

This subfolder contains the notebooks necessary to recreate the preliminary experiments with initializing LinTS models with small perturbations.

## Setup
In order to run the notebooks, please copy the `movielens_responses.csv`, `movielens_scaler.pkl`, and `movielens_users.csv` from the [table 2/3 directory](../../table_2_3).
You might also need to create an `output` directory within this directory.

## Notebooks
You can run the notebooks linked below in the environments they are named after to generate deterministic numbers.

- Run the [MacOS Big Sur MKL](MacOSBigSur_MKL.ipynb) notebook when using the [MacOS Big Sur MKL](../../environments/MacOSBigSur_MKL) environment.
- Run the [MacOS Big Sur OpenBLAS](MacOSBigSur_OpenBLAS.ipynb) notebook when using the [MacOS Big Sur OpenBLAS](../../environments/MacOSBigSur_OpenBLAS) environment.

Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment.
