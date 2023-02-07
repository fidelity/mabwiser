# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import Arm, Num, argmax, _BaseRNG, create_rng

SCALER_TOLERANCE = 1e-6


def fix_small_variance(scaler: StandardScaler) -> NoReturn:
    """
    Set variances close to zero to be equal to one in trained standard scaler to make computations stable.

    :param scaler: the scaler to check and fix variances for
    """
    if hasattr(scaler, 'scale_') and hasattr(scaler, 'var_'):
        # Get a mask to pull indices where std smaller than scaler_tolerance
        mask = scaler.scale_ <= SCALER_TOLERANCE

        # Fix standard deviation
        scaler.scale_[mask] = 1.0e+00

        # Fix variance accordingly. var_ is allowed to be 0 in scaler.
        # This helps to track if scale_ are set as ones due to zeros in variances.
        scaler.var_[mask] = 0.0e+00


class _RidgeRegression:

    def __init__(self, rng: _BaseRNG, alpha: Num = 1.0, l2_lambda: Num = 1.0, scale: bool = False):

        # Ridge Regression: https://onlinecourses.science.psu.edu/stat857/node/155/
        self.rng = rng                      # random number generator
        self.alpha = alpha                  # exploration parameter
        self.l2_lambda = l2_lambda          # regularization parameter
        self.scale = scale                  # scale contexts

        self.beta = None                    # (XtX + l2_lambda * I_d)^-1 * Xty = A^-1 * Xty
        self.A = None                       # (XtX + l2_lambda * I_d)
        self.A_inv = None                   # (XtX + l2_lambda * I_d)^-1
        self.Xty = None
        self.scaler = None

    def init(self, num_features: int):
        # By default, assume that
        # A is the identity matrix and Xty is set to 0
        self.Xty = np.zeros(num_features)
        self.A = self.l2_lambda * np.identity(num_features)
        self.A_inv = self.A.copy()
        self.beta = np.dot(self.A_inv, self.Xty)
        self.scaler = StandardScaler() if self.scale else None

    def fit(self, X: np.ndarray, y: np.ndarray):

        # Scale
        if self.scaler is not None:
            X = X.astype('float64')
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(X)
            else:
                self.scaler.partial_fit(X)
            fix_small_variance(self.scaler)
            X = self.scaler.transform(X)

        # X transpose
        Xt = X.T

        # Update A
        self.A = self.A + np.dot(Xt, X)
        self.A_inv = np.linalg.inv(self.A)

        # Add new Xty values to old
        self.Xty = self.Xty + np.dot(Xt, y)

        # Recalculate beta coefficients
        self.beta = np.dot(self.A_inv, self.Xty)

    def predict(self, x: np.ndarray):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Calculate default expectation y = x * b
        return np.dot(x, self.beta)

    def _scale_predict_context(self, x: np.ndarray):
        if not hasattr(self.scaler, 'scale_'):
            return x

        # Transform and return to previous shape. Convert to float64 to suppress any type warnings.
        return self.scaler.transform(x.astype('float64'))


class _LinTS(_RidgeRegression):

    def predict(self, x: np.ndarray):
        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Randomly sample coefficients from multivariate normal distribution
        # Covariance is enhanced with the exploration factor
        # Generates  random samples for all contexts in one single go. type(beta_sampled): np.ndarray
        beta_sampled = self.rng.multivariate_normal(self.beta, np.square(self.alpha) * self.A_inv, size=x.shape[0])

        # Calculate expectation y = x * beta_sampled
        return np.sum(x * beta_sampled, axis=1)


class _LinUCB(_RidgeRegression):

    def predict(self, x: np.ndarray):
        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Calculating x_A_inv
        x_A_inv = np.dot(x, self.A_inv)

        # Upper confidence bound = alpha * sqrt(x A^-1 xt). Notice that, x = xt
        # ucb values are claculated for all the contexts in one single go. type(ucb): np.ndarray
        ucb = self.alpha * np.sqrt(np.sum(x_A_inv * x, axis=1))

        # Calculate linucb expectation y = x * b + ucb
        return np.dot(x, self.beta) + ucb


class _Linear(BaseMAB):
    factory = {"ts": _LinTS, "ucb": _LinUCB, "ridge": _RidgeRegression}

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 alpha: Num, epsilon: Num, l2_lambda: Num, regression: str, scale: bool):
        super().__init__(rng, arms, n_jobs, backend)
        self.alpha = alpha
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda
        self.regression = regression
        self.scale = scale
        self.num_features = None

        # Create regression model for each arm
        self.arm_to_model = dict((arm, _Linear.factory.get(regression)(rng, alpha, l2_lambda, scale)) for arm in arms)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Initialize each model by arm
        self.num_features = contexts.shape[1]
        for arm in self.arms:
            self.arm_to_model[arm].init(num_features=self.num_features)

        # Reset warm started arms
        self._reset_arm_to_status()

        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=False)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=True)

    def predict(self, contexts: np.ndarray = None) -> Union[Arm, List[Arm]]:
        # Return predict for the given context
        return self._vectorized_predict_context(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        # Return predict expectations for the given context
        return self._vectorized_predict_context(contexts, is_predict=False)

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        self._warm_start(arm_to_features, distance_quantile)

    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_model[cold_arm] = deepcopy(self.arm_to_model[warm_arm])

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None):

        # Add to untrained_arms arms
        self.arm_to_model[arm] = _Linear.factory.get(self.regression)(self.rng, self.alpha, self.l2_lambda, self.scale)

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
        pass

    def _vectorized_predict_context(self, contexts: np.ndarray, is_predict: bool) -> List:

        # Converting the arms list to numpy array
        arms = deepcopy(self.arms)
        arms = np.array(arms)

        # Initializing array with expectations for each arm
        num_contexts = contexts.shape[0]
        arm_expectations = np.empty((num_contexts, len(arms)), dtype=float)

        # With epsilon probability, assign random flag to context
        random_values = self.rng.rand(num_contexts)
        random_mask = np.array(random_values < self.epsilon)
        random_indices = random_mask.nonzero()[0]

        # For random indices, generate random expectations
        arm_expectations[random_indices] = self.rng.rand((random_indices.shape[0], len(arms)))

        # For non-random indices, get expectations for each arm
        nonrandom_indices = np.where(~random_mask)[0]
        nonrandom_context = contexts[nonrandom_indices]
        arm_expectations[nonrandom_indices] = np.array([self.arm_to_model[arm].predict(nonrandom_context)
                                                        for arm in arms]).T

        if is_predict:
            predictions = arms[np.argmax(arm_expectations, axis=1)].tolist()
        else:
            predictions = [dict(zip(self.arms, value)) for value in arm_expectations]

        return predictions if len(predictions) > 1 else predictions[0]

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        self.arm_to_model.pop(arm)
