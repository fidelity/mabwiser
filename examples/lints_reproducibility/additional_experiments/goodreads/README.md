# Goodreads Reproducibility

This subfolder contains the notebooks necessary to recreate the reproducibility issues with LinTS on Goodreads data.

## Steps
Execute the below steps 1-3 with any of the `MacOS Big Sur MKL`, `MacOS Big Sur OpenBLAS`,  `Linux Ubuntu MKL`, or `Linux Ubuntu OpenBLAS` environments.

1. Download the young adult portion of the Goodreads dataset here [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home). The files you want to grab are [goodreads_books_young_adult.json.gz](https://drive.google.com/uc?id=1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol), [goodreads_interactions_young_adult.json.gz](https://drive.google.com/uc?id=1NNX7SWcKahezLFNyiW88QFPAqOAYP5qg), and
[goodreads_reviews_young_adult.json.gz](https://drive.google.com/uc?id=1M5iqCZ8a7rZRtsmY5KQ5rYnP9S0bQJVo). Place the datasets in the same directory as this readme.
2. Run through the [preprocessing notebook](Goodreads%20Preprocessing.ipynb). This will create the response matrix file `responses.csv.gz` and user features file `user_features.csv.gz`.
3. Run through the [sampling notebook](Goodreads%20Samples.ipynb) to sample from the generated files above. This will the create the sample response matrix file `sample_responses.csv.gz` and sample user features file `sample_user_features.csv.gz`.
4. Run through the appropriate reproducibility notebook based on your environment to generate train a LinTS model and get recommendations for the test users.
   - Run the [Linux Ubuntu MKL](LinuxUbuntu_MKL.ipynb) notebook when using the [Linux Ubuntu MKL](../../environments/LinuxUbuntu_MKL) environment.
   - Run the [Linux Ubuntu OpenBLAS](LinuxUbuntu_OpenBLAS.ipynb) notebook when using the [Linux Ubuntu OpenBLAS](../../environments/LinuxUbuntu_OpenBLAS) environment.
   - Run the [MacOS Big Sur MKL](MacOSBigSur_MKL.ipynb) notebook when using the [MacOS Big Sur MKL](../../environments/MacOSBigSur_MKL) environment.
   - Run the [MacOS Big Sur OpenBLAS](MacOSBigSur_OpenBLAS.ipynb) notebook when using the [MacOS Big Sur OpenBLAS](../../environments/MacOSBigSur_OpenBLAS) environment.
5. Once the notebooks are run, the [Analysis](Analysis.ipynb) notebook can be run in **any** environment to show the results.


## Results
The results of step 5 show that when using the legacy `RandomState` class or the new `Generator` class with default values (meaning SVD decomposition is used), the recommendations generated for the same user is different across the two environments. It also shows that using Cholesky decomposition with the new `Generator` class alleviates this problem and makes the recommendations reproducible across the two environments. This is in line with our hypothesis.
