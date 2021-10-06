# Linux Ubuntu OpenBLAS Environment

This readme sets up the Linux Ubuntu OpenBLAS environment required to run the notebooks named `LinuxUbuntu_OpenBLAS.ipynb`.

## Requirements
1. Linux Ubuntu 18.04.5. The exact version used in the notebooks is `Linux-4.15.0-151-generic-x86_64-with-debian-buster-sid`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n linux_ubuntu_openblas python=3.7.10
conda activate linux_ubuntu_openblas
pip install numpy==1.18.5
pip install scikit-learn==0.24.1
pip install tensorflow==1.15.5
pip install pandas
pip install mabwiser
pip install jupyter
```

These steps correctly install numpy with OpenBLAS backend on Windows. You can check the NumPy backend using:
```
import numpy as np
np.show_config()
```
This should not show any entries related to MKL.

## Running Notebooks
Once installed, you can run the relevant notebooks by using the following commands and navigating to the notebook you want to run:

```bash
conda activate linux_ubuntu_openblas
jupyter notebook
```
