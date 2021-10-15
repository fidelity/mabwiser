# MacOS Darwin OpenBLAS Environment

This readme sets up the MacOS Darwin OpenBLAS environment required to run the notebooks named `MacOSDarwin_OpenBLAS.ipynb`.

## Requirements
1. MacOS Darwin. The exact version used in the notebooks is `macOS-10.15.7-x86_64-i386-64bit`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n mac_dar_openblas python=3.8
conda activate mac_dar_openblas
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
conda activate mac_dar_openblas
jupyter notebook
```
