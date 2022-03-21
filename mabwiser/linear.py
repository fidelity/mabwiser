# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.configs.arm import ArmConfig
from mabwiser.utilities.general import argmax
from mabwiser.utilities.random import _BaseRNG, create_rng


class _RidgeRegression(ABC):
    def __init__(
        self,
        rng: _BaseRNG,
        alpha: float,
        l2_lambda: float,
        scaler: Optional[Callable] = None,
    ):
        # Ridge Regression: https://onlinecourses.science.psu.edu/stat857/node/155/
        self.rng = rng  # random number generator
        self.alpha = alpha  # exploration parameter
        self.l2_lambda = l2_lambda  # regularization parameter
        self.scaler = scaler  # standard scaler object

        self.beta = None  # (XtX + l2_lambda * I_d)^-1 * Xty = A^-1 * Xty
        self.A = None  # (XtX + l2_lambda * I_d)
        self.A_inv = None  # (XtX + l2_lambda * I_d)^-1
        self.Xty = None

    def init(self, num_features: int) -> None:
        # By default, assume that
        # A is the identity matrix and Xty is set to 0
        self.Xty = np.zeros(num_features)
        self.A = self.l2_lambda * np.identity(num_features)
        self.A_inv = self.A.copy()
        self.beta = np.dot(self.A_inv, self.Xty)

    def fit(self, x: np.ndarray, y) -> None:

        # Scale
        if self.scaler is not None:
            x = self.scaler(x.astype("float64"))

        # X transpose
        xt = x.T

        # Update A
        self.A = self.A + np.dot(xt, x)
        self.A_inv = np.linalg.inv(self.A)

        # Add new Xty values to old
        self.Xty = self.Xty + np.dot(xt, y)

        # Recalculate beta coefficients
        self.beta = np.dot(self.A_inv, self.Xty)

    @abstractmethod
    def predict(self, x):
        pass

    def _scale_predict_context(self, x: np.ndarray) -> np.ndarray:
        # Reshape 1D array to 2D
        x = x.reshape(1, -1)

        # Transform and return to previous shape. Convert to float64 to suppress any type warnings.
        return self.scaler(x.astype("float64")).reshape(-1)


class _LinGreedyBase(_RidgeRegression):
    def __init__(
        self,
        rng: _BaseRNG,
        alpha: float,
        l2_lambda: float,
        scaler: Optional[Callable] = None,
    ):
        super(_LinGreedyBase, self).__init__(
            rng=rng, alpha=alpha, l2_lambda=l2_lambda, scaler=scaler
        )

    def predict(self, x):
        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Calculate default expectation y = x * b
        return np.dot(x, self.beta)


class _LinTSBase(_RidgeRegression):
    def __init__(
        self,
        rng: _BaseRNG,
        alpha: float,
        l2_lambda: float,
        scaler: Optional[Callable] = None,
    ):
        super(_LinTSBase, self).__init__(
            rng=rng, alpha=alpha, l2_lambda=l2_lambda, scaler=scaler
        )

    def predict(self, x):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Randomly sample coefficients from multivariate normal distribution
        # Covariance is enhanced with the exploration factor
        beta_sampled = self.rng.multivariate_normal(
            self.beta, np.square(self.alpha) * self.A_inv
        )

        # Calculate expectation y = x * beta_sampled
        return np.dot(x, beta_sampled)


class _LinUCBBase(_RidgeRegression):
    def __init__(
        self,
        rng: _BaseRNG,
        alpha: float,
        l2_lambda: float,
        scaler: Optional[Callable] = None,
    ):
        super(_LinUCBBase, self).__init__(
            rng=rng, alpha=alpha, l2_lambda=l2_lambda, scaler=scaler
        )

    def predict(self, x):

        # Scale
        if self.scaler is not None:
            x = self._scale_predict_context(x)

        # Upper confidence bound = alpha * sqrt(x A^-1 xt). Notice that, x = xt
        ucb = self.alpha * np.sqrt(np.dot(np.dot(x, self.A_inv), x))

        # Calculate linucb expectation y = x * b + ucb
        return np.dot(x, self.beta) + ucb


class _Linear(BaseMAB):
    def __init__(
        self,
        model: Union[Type["_LinGreedyBase"], Type["_LinTSBase"], Type["_LinUCBBase"]],
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        alpha: float,
        epsilon: float,
        l2_lambda: float,
        backend: Optional[str] = None,
        arm_to_scaler: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend)
        self.alpha = alpha
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda
        self.model = model

        # Create ridge regression model for each arm
        self.num_features = None

        if arm_to_scaler is None:
            arm_to_scaler = dict((arm, None) for arm in arms)

        self.arm_to_model = dict(
            (arm, self.model(rng, alpha, l2_lambda, arm_to_scaler[arm])) for arm in arms
        )

    def fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Initialize each model by arm
        self.num_features = contexts.shape[1]
        for arm in self.arms:
            self.arm_to_model[arm].init(num_features=self.num_features)

        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[List, str]:
        # Return predict for the given context
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(
        self, contexts: Optional[np.ndarray] = None
    ) -> Union[List, str]:
        # Return predict expectations for the given context
        return self._parallel_predict(contexts, is_predict=False)

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_model[cold_arm] = deepcopy(self.arm_to_model[warm_arm])

    def _uptake_new_arm(self, arm: ArmConfig) -> None:

        # Add to untrained_arms arms
        self.arm_to_model[arm.arm] = self.model(
            self.rng, self.alpha, self.l2_lambda, arm.scaler
        )

        # If fit happened, initialize the new arm to defaults
        is_fitted = self.num_features is not None
        if is_fitted:
            self.arm_to_model[arm.arm].init(num_features=self.num_features)

    def _fit_arm(
        self,
        arm: str,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ):

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

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:

        # Get local copy of model, arm_to_expectation and arms to minimize
        # communication overhead between arms (processes) using shared objects
        arm_to_model = deepcopy(self.arm_to_model)
        arm_to_expectation = deepcopy(self.arm_to_expectation)
        arms = deepcopy(self.arms)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)
        for index, row in enumerate(contexts):
            # Each row needs a separately seeded rng for reproducibility in parallel
            rng = create_rng(seed=seeds[index])

            # With epsilon probability set arm expectation to random value
            if rng.rand() < self.epsilon:
                for arm in arms:
                    arm_to_expectation[arm] = rng.rand()

            else:
                # Create new seeded generator for model to ensure reproducibility
                model_rng = create_rng(seed=seeds[index])
                for arm in arms:
                    arm_to_model[arm].rng = model_rng

                    # Get the expectation of each arm from its trained model
                    arm_to_expectation[arm] = arm_to_model[arm].predict(row)

            if is_predict:
                predictions[index] = argmax(arm_to_expectation)
            else:
                predictions[index] = arm_to_expectation.copy()

        # Return list of predictions
        return predictions

    def _drop_existing_arm(self, arm: str) -> None:
        self.arm_to_model.pop(arm)


class _LinGreedy(_Linear):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        epsilon: float,
        l2_lambda: float,
        alpha: float = 0.0,
        backend: Optional[str] = None,
        arm_to_scaler: Optional[Dict[str, Callable]] = None,
    ):
        super(_LinGreedy, self).__init__(
            model=_LinGreedyBase,
            rng=rng,
            arms=arms,
            n_jobs=n_jobs,
            alpha=alpha,
            epsilon=epsilon,
            l2_lambda=l2_lambda,
            backend=backend,
            arm_to_scaler=arm_to_scaler,
        )


class _LinTS(_Linear):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        alpha: float,
        epsilon: float,
        l2_lambda: float,
        backend: Optional[str] = None,
        arm_to_scaler: Optional[Dict[str, Callable]] = None,
    ):
        super(_LinTS, self).__init__(
            model=_LinTSBase,
            rng=rng,
            arms=arms,
            n_jobs=n_jobs,
            alpha=alpha,
            epsilon=epsilon,
            l2_lambda=l2_lambda,
            backend=backend,
            arm_to_scaler=arm_to_scaler,
        )


class _LinUCB(_Linear):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        alpha: float,
        epsilon: float,
        l2_lambda: float,
        backend: Optional[str] = None,
        arm_to_scaler: Optional[Dict[str, Callable]] = None,
    ):
        super(_LinUCB, self).__init__(
            model=_LinUCBBase,
            rng=rng,
            arms=arms,
            n_jobs=n_jobs,
            alpha=alpha,
            epsilon=epsilon,
            l2_lambda=l2_lambda,
            backend=backend,
            arm_to_scaler=arm_to_scaler,
        )
