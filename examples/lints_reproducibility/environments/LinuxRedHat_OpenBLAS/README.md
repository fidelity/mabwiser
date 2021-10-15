# Linux RedHat OpenBLAS Environment

This readme sets up the Linux RedHat OpenBLAS environment required to run the notebooks named `LinuxRedHat_OpenBLAS.ipynb`.

## Requirements
1. Linux RedHat. The exact version used in the notebooks is `Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-glibc2.10`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n linux_rh python=3.8
conda activate linux_rh
pip install -r requirements.txt
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
conda activate linux_rh
jupyter notebook
```
