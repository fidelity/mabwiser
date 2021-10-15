# Windows OpenBLAS Environment

This readme sets up the Windows OpenBLAS environment required to run the notebooks named `Windows_OpenBLAS.ipynb`.

## Requirements
1. Windows 10. The exact version used in the notebooks is `Windows-10-10.0.18362-SP0`, results might or might not be reproducible with a different version.
2. [Anaconda](https://www.anaconda.com/)

## Installation
From the anaconda prompt, run the following commands:

```bash
conda create -n windows python=3.7.3
conda activate windows
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
conda activate windows
jupyter notebook
```
