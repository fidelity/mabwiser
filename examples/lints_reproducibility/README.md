# LinTS Reproducibility
This directory contains all the source code necessary to verify the reproducibility of LinTS as presented in the paper ["Non-Deterministic Behavior of Thompson Sampling with Linear Payoffs and How to Avoid It"](https://openreview.net/forum?id=sX9d3gfwtE).

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

## Incidence Rate on GitHub

Given the popularity of the NumPy library and the SVD algorithm in the ML community, we expect that this issue is prevalent in many research and industrial applications.
While it is not trivial to know the full extent of the impact of this non-determinism, keyword searches on GitHub can serve to give an indication of how pervasive this problem is in the open source community.
For this, we looked at the difference between the old NumPy random number generator, which can only use SVD for its multivariate random sampling, and the new random generator, which can use Cholesky.
At the time of writing, we can find 49K instances of `np.random.multivariate_random` calls (which will have this determinism issue) on [GitHub](https://github.com/search?q=np.random.multivariate_normal&type=code), 76K instances of `np.random` followed by `multivariate_random` at some point (which contains calls made to both the old random number generator and the new one) ([GitHub](https://github.com/search?q=np.random++.multivariate_normal&type=Code)), and 2.7K instances of `np.random.default_rng(…).multivariate_random`, which are calls made to the new version of numpy random number generator ([GitHub](https://github.com/search?q=np.random.default_rng++.multivariate_normal&type=code)).
Similarly, while we can find [343K instances](https://github.com/search?q=np.random.RandomState&type=Code) of the old random number generator, we can only find [29K](https://github.com/search?q=np.random.default_rng&type=Code) instances of the new random number generator.
While these queries cannot provide exhaustive numbers, it is for certain that there will be reproducibility issues for some of the 49K calls to the `np.random.multivariate_random` function.

## Citation

If you use this reproducibility analysis in a publication, please cite it as:

**[TMLR 2022]** [D. Kilitcioglu and S. Kadioglu, "Non-Deterministic Behavior of Thompson Sampling with Linear Payoffs and How to Avoid It"](https://openreview.net/forum?id=sX9d3gfwtE)

```bibtex
    @article{
      kilitcioglu2022nondeterministic,
      title={Non-Deterministic Behavior of Thompson Sampling with Linear Payoffs and How to Avoid It},
      author={Doruk Kilitcioglu and Serdar Kadioglu},
      journal={Transactions on Machine Learning Research},
      year={2022},
      url={https://openreview.net/forum?id=sX9d3gfwtE},
      note={Reproducibility Certification}
    }
```
