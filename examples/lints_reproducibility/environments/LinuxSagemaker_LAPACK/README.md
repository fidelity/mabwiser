# Linux Sagemaker LAPACK Environment

This readme sets up the Linux Sagemaker LAPACK environment required to run the notebooks named `LinuxSagemaker_LAPACK.ipynb`.

## Requirements
1. Linux Sagemaker. The exact version used in the notebooks is `Linux-4.14.225-121.362.amzn1.x86_64-x86_64-with-glibc2.9`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n linux_sag python=3.8
conda activate linux_sag
pip install -r requirements.txt
```

These steps correctly install numpy with OpenBLAS backend on Windows. You can check the NumPy backend using:
```
import numpy as np
np.show_config()
```
This should not show any entries related to MKL and should show LAPACK related entries.

## Running Notebooks
Once installed, you can run the relevant notebooks by using the following commands and navigating to the notebook you want to run:

```bash
conda activate linux_sag
jupyter notebook
```
