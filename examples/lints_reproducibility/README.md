# LinTS Reproducibility
This directory contains all the code necessary to evaluate the reproducibility of LinTS. The most up-to-date code can be found [online](https://github.com/fidelity/mabwiser/tree/master/examples/lints_reproducibility).

## Environments
Each notebook that is provided in this repo relies on one of the environments described in the [environments](environments) folder.
In order to run a notebook, you have to follow the installation instructions for the environment you wish to run.
To ensure correctness of the results, you should have the same platform as the notebook you're running.
The only exception to this are the analysis notebooks, named `Analysis.ipynb`, which can be run under any environment.

Unfortunately the only way to reliably reproduce all the results is to own multiple computers or have access to all the Operating Systems mentioned in the environments.
Even using matching backends (OpenBLAS vs MKL) isn't enough.
This reproducibility issue forms the basis of our contribution, as even though the non-Cholesky implementations suffer from this problem, using Cholesky decomposition leads to the same results across all environments.

## Experiment Reproducibility
- [Table 2 & 3](table_2_3) folder shows the experiments run to generate tables 2 and 3, which demonstrate the reproducibility issues on the MovieLens data set across 5 different environments.
- [Table 6](table_6) folder shows the reproducibility issues with the [Deep Bayesian Bandits library](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits).
