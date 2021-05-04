# LinTS Reprodubility

## Computational Reproducibility Notebooks
TODO - Emily

## General Reproducibility Notebooks
- [Multivariate Normal Reproducibility notebook](Multivariate%20Normal%20Reproducibility.ipynb) shows the reproducibility issue with sampling from the multivariate normal distribution in general.
- [LinTS MovieLens Recommendations notebook](LinTS%20Movie%20Recommendations.ipynb) shows the reproducibility issues on MovieLens dataset across 4 different environments.
- [Deep Bayesian Bandits Reproducibility notebook](DBB%20Reproducibility.ipynb) shows the reproducibility issues with the [Deep Bayesian Bandits library](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits).

## Data
- `movielens_responses.csv` contains the cleaned-up response matrix generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- `movielens_users.csv` contains the cleaned-up user features generated from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- `simulated_data.csv` contains the simulated data with 4 arms
- `mushroom.data` contains the Mushroom dataset used in Deep Bayesian Bandits reproducibility notebook, downloaded from the links in the [Deep Bayesian Bandits Readme](https://github.com/tensorflow/models/tree/36101ab4095065a4196ff4f6437e94f0d91df4e9/research/deep_contextual_bandits)

## Misc
- `*.pkl` files contain various MAB models and NumPy arrays generated as part of the notebooks above
- `example_ssh_config.json` shows the config necessary to run some of the experiments over ssh in the general reproducibility notebooks
- `utils.py` contains minor utilities for ensuring correctness
