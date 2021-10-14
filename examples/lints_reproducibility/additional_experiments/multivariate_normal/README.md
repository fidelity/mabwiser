# Multivariate Normal Sampling Reproducibility

This subfolder contains the notebooks necessary to recreate the reproducibility issues with Multivariate Normal Sampling in NumPy.
The experiments show the reproducibility issue when sampling from a multivariate normal distribution.

## Notebooks
You can run the notebooks linked below in the environments they are named after to generate deterministic numbers.

- Run the [Linux Ubuntu MKL](LinuxUbuntu_MKL.ipynb) notebook when using the [Linux Ubuntu MKL](../../environments/LinuxUbuntu_MKL) environment.
- Run the [Linux Ubuntu OpenBLAS](LinuxUbuntu_OpenBLAS.ipynb) notebook when using the [Linux Ubuntu OpenBLAS](../../environments/LinuxUbuntu_OpenBLAS) environment.
- Run the [MacOS Big Sur MKL](MacOSBigSur_MKL.ipynb) notebook when using the [MacOS Big Sur MKL](../../environments/MacOSBigSur_MKL) environment.
- Run the [MacOS Big Sur OpenBLAS](MacOSBigSur_OpenBLAS.ipynb) notebook when using the [MacOS Big Sur OpenBLAS](../../environments/MacOSBigSur_OpenBLAS) environment.

Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment.
