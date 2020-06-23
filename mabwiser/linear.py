# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import Arm, Num, argmax, _BaseRNG


class _RidgeRegression:

    def __init__(self, rng: _BaseRNG, l2_lambda: Num = 1.0, alpha: Num = 1.0,
                 scaler: Optional[Callable] = None):

        # Ridge Regression: https://onlinecourses.science.psu.edu/stat857/node/155/
        self.rng = rng                      # random number generator
        self.l2_lambda = l2_lambda          # regularization parameter
        self.alpha = alpha                  # exploration parameter
        self.scaler = scaler                # standard scaler object

        self.beta = None                    # (XtX + l2_lambda * I_d)^-1 * Xty = A^-1 * Xty
        self.A = None                       # (XtX + l2_lambda * I_d)
        self.A_inv = None                   # (XtX + l2_lambda * I_d)^-1
        self.Xty = None

    def init(self, num_features):
        # By default, assume that
        # A is the identity matrix and Xty is set to 0
        self.Xty = np.zeros(num_features)
        self.A = self.l2_lambda * np.identity(num_features)
        self.A_inv = self.A.copy()
        self.beta = np.dot(self.A_inv, self.Xty)

    def fit(self, X, y):

        # Scale
        if self.scaler is not None:
            X = self.scaler.transform(X.astype('float64'))

        # X transpose
        Xt = X.T

        # Update A
        self.A = self.A + np.dot(Xt, X)
        self.A_inv = np.linalg.inv(self.A)

        # Add new Xty values to old
        self.Xty = self.Xty + np.dot(Xt, y)

        # Recalculate beta coefficients
        self.beta = np.dot(self.A_inv, self.Xty)

    def predict(self, x):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Calculate default expectation y = x * b
        return np.dot(x, self.beta)

    def _scale_predict_context(self, x):
        # Reshape 1D array to 2D
        x = x.reshape(1, -1)

        # Transform and return to previous shape. Convert to float64 to suppress any type warnings.
        return self.scaler.transform(x.astype('float64')).reshape(-1)


class _LinTS(_RidgeRegression):

    def __init__(self, rng: _BaseRNG, l2_lambda: Num = 1.0, alpha: Num = 1.0,
                 scaler: Optional[Callable] = None):
        super().__init__(rng, l2_lambda, alpha, scaler)

        # Set covariance to none
        self.covar_decomposed = None

    def init(self, num_features):
        super().init(num_features)

        # Calculate covariance
        self.covar_decomposed = self._cholesky()

    def fit(self, X, y):
        super().fit(X, y)

        # Calculate covariance
        self.covar_decomposed = self._cholesky()

    def predict(self, x):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Multivariate Normal Sampling
        # Adapted from the implementation in numpy.random.generator.multivariate_normal version 1.18.0
        # Uses the cholesky implementation from numpy.linalg instead of numpy.dual
        sampled_norm = self.rng.standard_normal(self.beta.shape[0])

        # Randomly sample coefficients from Normal Distribution N(mean=beta, std=covar_decomposed)
        beta_sampled = self.beta + np.dot(sampled_norm, self.covar_decomposed)

        # Calculate expectation y = x * beta_sampled
        return np.dot(x, beta_sampled)

    def _cholesky(self):

        # Calculate exploration factor
        exploration = np.square(self.alpha) * self.A_inv

        # Covariance using cholesky decomposition
        covar_decomposed = np.linalg.cholesky(exploration)

        return covar_decomposed


class _LinUCB(_RidgeRegression):

    def predict(self, x):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Upper confidence bound = alpha * sqrt(x A^-1 xt). Notice that, x = xt
        ucb = (self.alpha * np.sqrt(np.dot(np.dot(x, self.A_inv), x)))

        # Calculate linucb expectation y = x * b + ucb
        return np.dot(x, self.beta) + ucb


class _Linear(BaseMAB):

    factory = {"ts": _LinTS, "ucb": _LinUCB, "ridge": _RidgeRegression}

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 l2_lambda: Num, alpha: Num, regression: str, arm_to_scaler: Optional[Dict[Arm, Callable]] = None):
        super().__init__(rng, arms, n_jobs, backend)
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.regression = regression

        # Create ridge regression model for each arm
        self.num_features = None

        if arm_to_scaler is None:
            arm_to_scaler = dict((arm, None) for arm in arms)

        self.arm_to_model = dict((arm, _Linear.factory.get(regression)(rng, l2_lambda,
                                                                       alpha, arm_to_scaler[arm])) for arm in arms)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Initialize each model by arm
        self.num_features = contexts.shape[1]
        for arm in self.arms:
            self.arm_to_model[arm].init(num_features=self.num_features)

        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: np.ndarray = None):
        # Return predict for the given context
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None):
        # Return predict expectations for the given context
        return self._parallel_predict(contexts, is_predict=False)

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        # Add to untrained_arms arms
        self.arm_to_model[arm] = _Linear.factory.get(self.regression)(self.rng, self.l2_lambda, self.alpha, scaler)

        # If fit happened, initialize the new arm to defaults
        is_fitted = self.num_features is not None
        if is_fitted:
            self.arm_to_model[arm].init(num_features=self.num_features)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        # Get local copy of model to minimize communication overhead
        # between arms (processes) using shared object
        lr = deepcopy(self.arm_to_model[arm])

        # Skip the arms with no data
        indices = np.where(decisions == arm)
        if indices[0].size == 0:
            return lr

        # Fit the regression
        X = contexts[indices]
        y = rewards[indices]
        lr.fit(X, y)

        self.arm_to_model[arm] = lr

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Get local copy of model, arm_to_expectation and arms to minimize
        # communication overhead between arms (processes) using shared objects
        arm_to_model = deepcopy(self.arm_to_model)
        arm_to_expectation = deepcopy(self.arm_to_expectation)
        arms = deepcopy(self.arms)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)
        for index, row in enumerate(contexts):
            # Each row needs a separately seeded rng for reproducibility in parallel
            rng = np.random.RandomState(seed=seeds[index])

            for arm in arms:
                # Copy the row rng to the deep copied model in arm_to_model
                arm_to_model[arm].rng = rng

                # Get the expectation of each arm from its trained model
                arm_to_expectation[arm] = arm_to_model[arm].predict(row)

            if is_predict:
                predictions[index] = argmax(arm_to_expectation)
            else:
                predictions[index] = arm_to_expectation.copy()

        # Return list of predictions
        return predictions
