# MacOS Darwin MKL Environment

This readme sets up the MacOS Darwin MKL environment required to run the notebooks named `MacOSDarwin_MKL.ipynb`.

## Requirements
1. MacOS Darwin. The exact version used in the notebooks is `Darwin-19.5.0-x86_64-i386-64bit`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n mac_dar_mkl python=3.8
conda activate mac_dar_mkl
conda install numpy==1.18.1
conda install scikit-learn==0.24.1
conda install jupyter
pip install pandas
pip install mabwiser
pip install jurity
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
conda activate mac_dar_mkl
jupyter notebook
```
