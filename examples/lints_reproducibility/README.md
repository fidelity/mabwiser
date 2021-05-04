# LinTS Reprodubility
This directory contains all the code necessary to evaluate the reproducibility of LinTS. The most up-to-date code can be found [online](https://github.com/fidelity/mabwiser/tree/master/examples/lints_reproducibility).


## General Reproducibility Notebooks
- [Multivariate_Normal_Reproducibility notebook](Multivariate_Normal_Reproducibility.ipynb) shows the reproducibility issue with sampling from the multivariate normal distribution in general.
- [LinTS_Minimal notebook](LinTS_Minimal.ipynb) shows the reproducibility issue with our minimal example in 5 different environments.

## Experiment Reproducibility Notebooks
- [LinTS_Simulated_Data_Bandit notebook](LinTS_Simulated_Data_Bandit.ipynb) shows the experiments run to demonstrate the reproducibility issues on a simulated data set across 5 different environments.
- [LinTS_Simulated_Data_Cholesky_Bandit notebook](LinTS_Simulated_Data_Cholesky_Bandit.ipynb) shows the experiments run to demonstrate reproducibility with Cholesky decomposition on the simulated data set across 5 different environments.
- [LinTS_Simulated_Data_Analysis notebook](LinTS_Simulated_Data_Analysis.ipynb) shows the data analysis of the outputs from LinTS_Simulated_Data_Bandit and LinTS_Simulated_Data_Cholesky_Bandit.
- [LinTS_MovieLens_Bandit notebook](LinTS_MovieLens_Bandit.ipynb) shows the experiments run to demonstrate the reproducibility issues on MovieLens data set across 5 different environments.
- [LinTS_MovieLens_Cholesky_Bandit notebook](LinTS_MovieLens_Cholesky_Bandit.ipynb) shows the experiments run to demonstrate reproducibility with Cholesky decomposition on MovieLens data set across 5 different environments.
- [LinTS_MovieLens_Analysis notebook](LinTS_MovieLens_Analysis.ipynb) shows the data analysis of the outputs from LinTS_MovieLens_Bandit and LinTS_MovieLens_Cholesky_Bandit.
- [DBB_Reproducibility notebook](DBB_Reproducibility.ipynb) shows the reproducibility issues with the [Deep Bayesian Bandits library](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits).

## Data
- `movielens_responses.csv` contains the cleaned-up response matrix generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- `movielens_users.csv` contains the cleaned-up user features generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- `simulated_data.csv` contains the simulated data with 4 arms
- `mushroom.data` contains the Mushroom dataset used in Deep Bayesian Bandits reproducibility notebook, downloaded from the links in the [Deep Bayesian Bandits Readme](https://github.com/tensorflow/models/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/deep_contextual_bandits)

## Misc
- `*.pkl` files contain various MAB models and NumPy arrays generated as part of the notebooks above
- `example_ssh_config.json` shows the config necessary to run some of the experiments over ssh in the general reproducibility notebooks
- `utils.py` contains minor utilities for ensuring correctness

## Additional Experiments
- See [Goodreads](goodreads) folder for additional experimentation with Goodreads data.
