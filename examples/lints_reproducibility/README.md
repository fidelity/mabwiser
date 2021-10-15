# LinTS Reproducibility
This directory contains all the source code necessary to verify the reproducibility of LinTS as presented in the paper. 

The main contribution of our paper is to show that the Thompson Sampling with linear payofss (LinTS) algorithm suffers from non-deterministic behavior. This is highly problematic for experimental results and shadows our evaluations and conclusions. 

As a remedy, we further show that the the issue can be avoided by using Cholesky decomposition for multi-variate sampling. 

## Environments
Each notebook provided in this repo relies on a specific environment given in the [environments](environments) folder.
To run a notebook, you must follow the installation instructions for the corresponding environment.
Establishing the same platform is important to ensure correctness of the results. 
The only exception to this are the analysis notebooks, named `Analysis.ipynb`, which can be run under any environment.

It is unfortunate that the only way to reliably reproduce the results from the paper is to have access to these different environments and Operating Systems. Even using matching backends is not enough. For instance, OpenBLAS vs MKL can lead to subtle differences. This is exactly the basis of our comprehensive experimental study in the paper.

Thankfully, as we prove and demonstrate in the paper, using Cholesky decomposition leads to the same results across all environments.

## Experiments from the Paper
- [Table 2 & 3](table_2_3) folders demonstrate the reproducibility issues on the MovieLens data set across 5 different environments.
- [Table 4](table_4) folder demonstrates the reproducibility issues with the [Deep Bayesian Bandits library](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits).

## Additional Experiments
There are additional experiments, not presented in detail in the paper due to space limitations. These experiments further verify our study. 
- [Additional Experiments](additional_experiments) folder demonstrates reproducibility issues in the minimal example, a synthetic dataset, and the Goodreads dataset.
