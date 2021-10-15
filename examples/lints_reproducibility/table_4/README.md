# Table 4 - Deep Bayesian Bandits

This directory contains the notebooks required to create the data and metrics for table 6.
Table 4 shows how the reproducibility problem can be created in Google's [Deep Bayesian Bandits](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits) work.
We look at their LinTS implementation, which forms the baseline of their experiments.
We show how using Cholesky decomposition creates systematic and reproducible results.

## Set Up
In order to recreate this table, you will need to:
- Clone the [tensorflow models repo](https://github.com/tensorflow/models)
- Switch to the `archive` branch
- Copy `models/research/deep_contextual_bandits/bandits` folder to this directory.

## Notebooks
You can run the notebooks linked below in the environments they are named after to generate deterministic numbers.

- Run the [Linux Ubuntu MKL](LinuxUbuntu_MKL.ipynb) notebook when using the [Linux Ubuntu MKL](../environments/LinuxUbuntu_MKL) environment.
- Run the [Linux Ubuntu OpenBLAS](LinuxUbuntu_OpenBLAS.ipynb) notebook when using the [Linux Ubuntu OpenBLAS](../environments/LinuxUbuntu_OpenBLAS) environment.
- Run the [MacOS Big Sur MKL](MacOSBigSur_MKL.ipynb) notebook when using the [MacOS Big Sur MKL](../environments/MacOSBigSur_MKL) environment.
- Run the [MacOS Big Sur OpenBLAS](MacOSBigSur_OpenBLAS.ipynb) notebook when using the [MacOS Big Sur OpenBLAS](../environments/MacOSBigSur_OpenBLAS) environment.

Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment.
This notebook will generate the numbers used in table 6, in the appropriately marked locations of the notebook.

## Data
- `mushroom.data` contains the Mushroom dataset used in Deep Bayesian Bandits reproducibility notebook, downloaded from the links in the [Deep Bayesian Bandits Readme](https://github.com/tensorflow/models/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/deep_contextual_bandits)
