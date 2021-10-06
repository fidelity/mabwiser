# Linux Ubuntu MKL Environment

This readme sets up the Linux Ubuntu MKL environment required to run the notebooks named `LinuxUbuntu_MKL.ipynb`.

## Requirements
1. Linux Ubuntu 18.04.5. The exact version used in the notebooks is `Linux-4.15.0-151-generic-x86_64-with-debian-buster-sid`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n linux_ubuntu_mkl python=3.7.10
conda activate linux_ubuntu_mkl
conda install numpy==1.18.5
conda install scikit-learn==0.24.1
conda install jupyter
pip install tensorflow==1.15.5
pip install pandas
pip install mabwiser
```

These steps correctly install numpy with MKL backend on MacOS. You can check the NumPy backend using:
```
import numpy as np
np.show_config()
```
This should show some entries related to MKL.

## Running Notebooks
Once installed, you can run the relevant notebooks by using the following commands and navigating to the notebook you want to run:

```bash
conda activate linux_ubuntu_mkl
jupyter notebook
```
