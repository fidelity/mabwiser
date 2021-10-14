# Goodreads Reproducibility

This subfolder contains the notebooks necessary to recreate the reproducibility issues with LinTS on Goodreads data.

## Steps
Execute the below steps 1-3 with any of the `MacOS Big Sur MKL`, `MacOS Big Sur OpenBLAS`,  `Linux Ubuntu MKL`, or `Linux Ubuntu OpenBLAS` environments.

1. Download the young adult portion of the Goodreads dataset here [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home). The files you want to grab are [goodreads_books_young_adult.json.gz](https://drive.google.com/uc?id=1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol), [goodreads_interactions_young_adult.json.gz](https://drive.google.com/uc?id=1NNX7SWcKahezLFNyiW88QFPAqOAYP5qg), and
[goodreads_reviews_young_adult.json.gz](https://drive.google.com/uc?id=1M5iqCZ8a7rZRtsmY5KQ5rYnP9S0bQJVo). Place the datasets in the same directory as this readme.
2. Run through the [preprocessing notebook](Goodreads%20Preprocessing.ipynb). This will create the response matrix file `responses.csv.gz` and user features file `user_features.csv.gz`.
3. Run through the [sampling notebook](Goodreads%20Samples.ipynb) to sample from the generated files above. This will the create the sample response matrix file `sample_responses.csv.gz` and sample user features file `sample_user_features.csv.gz`.
4. Run through the [reproducibility notebook](LinTS%20Goodreads%20Recommendations.ipynb) to generate train a LinTS model and get recommendations for the test users. Observe the results.


## Misc
- `example_ssh_config.json` shows the config necessary to run some of the experiments over ssh in the general reproducibility notebooks


## Results
The results of step 4 show that when using the legacy `RandomState` class or the new `Generator` class with default values (meaning SVD decomposition is used), the recommendations generated for the same user is different across the two environments. It also shows that using Cholesky decomposition with the new `Generator` class alleviates this problem and makes the recommendations reproducible across the two environments. This is in line with our hypothesis.
