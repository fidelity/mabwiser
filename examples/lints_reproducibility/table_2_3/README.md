# Table 2 & 3 - LinTS Reproducibility on MovieLens Data

This directory contains the notebooks required to create the data and metrics for tables 2 and 3.
Tables 2 and 3 show how the reproducibility problem can be created using MovieLens recommendation data, and how using Cholesky decomposition creates systematic and reproducible results.

## Notebooks
You can run the notebooks linked below in the environments they are named after to generate deterministic numbers.

- Run the [Linux RedHat OpenBLAS](LinuxRedHat_OpenBLAS.ipynb) notebook when using the [Linux RedHat OpenBLAS](../environments/LinuxRedHat_OpenBLAS) environment.
- Run the [Linux Sagemaker LAPACK](LinuxSagemaker_LAPACK.ipynb) notebook when using the [Linux Sagemaker LAPACK](../environments/LinuxSagemaker_LAPACK) environment.
- Run the [MacOS Darwin MKL](MacOSDarwin_MKL.ipynb) notebook when using the [MacOS Darwin MKL](../environments/MacOSDarwin_MKL) environment.
- Run the [MacOS Darwin OpenBLAS](MacOSDarwin_OpenBLAS.ipynb) notebook when using the [MacOS Darwin OpenBLAS](../environments/MacOSDarwin_OpenBLAS) environment.
- Run the [Windows OpenBLAS](Windows_OpenBLAS.ipynb) notebook when using the [Windows OpenBLAS](../environments/Windows_OpenBLAS) environment.

Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment.
This notebook will generate the numbers used in tables 2 & 3, in the appropriately marked locations of the notebook.

## Data
- `movielens_responses.csv` contains the cleaned-up response matrix generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- `movielens_users.csv` contains the cleaned-up user features generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
