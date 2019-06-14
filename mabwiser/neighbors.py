# -*- coding: utf-8 -*-
# SPDX-License-Identifer: Apache-2.0

from copy import deepcopy
from typing import Callable, List, NoReturn, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, Num, reset


class _Neighbors(BaseMAB):
    def __init__(self, rng: np.random.RandomState, arms: List[Arm], n_jobs: int,
                 lp: Union[_EpsilonGreedy, _Linear, _Random, _Softmax, _ThompsonSampling, _UCB1], metric: str):
        super().__init__(rng, arms, n_jobs)
        self.lp = lp
        self.metric = metric

        self.decisions = None
        self.rewards = None
        self.contexts = None

        # Initialize the arm expectations to nan
        # When there are neighbors, expectations of the underlying learning policy is used
        # When there are no neighbors, return nan expectations
        reset(self.arm_to_expectation, np.nan)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Set the historical data for prediction
        self.decisions = decisions
        self.contexts = contexts

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.lp.is_contextual_binarized = False
            self.rewards = self.lp._get_binary_rewards(decisions, rewards)
            self.lp.is_contextual_binarized = True
        else:
            self.rewards = rewards

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.lp.is_contextual_binarized = False
            rewards = self.lp._get_binary_rewards(decisions, rewards)
            self.lp.is_contextual_binarized = True

        # Add more historical data for prediction
        self.decisions = np.concatenate((self.decisions, decisions))
        self.contexts = np.concatenate((self.contexts, contexts))
        self.rewards = np.concatenate((self.rewards, rewards))

    def predict(self, contexts: np.ndarray = None):

        # Return predict within the neighborhood
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None):

        # Return predict expectations within the neighborhood
        return self._parallel_predict(contexts, is_predict=False)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        """Abstract method to be implemented by child classes."""
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        """Abstract method to be implemented by child classes."""
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.lp.add_arm(arm, binarizer)


class _Radius(_Neighbors):

    def __init__(self, rng: np.random.RandomState, arms: List[Arm], n_jobs: int,
                 lp: Union[_EpsilonGreedy, _Softmax, _ThompsonSampling, _UCB1, _Linear],
                 radius: Num, metric: str):
        super().__init__(rng, arms, n_jobs, lp, metric)

        self.radius = radius

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Copy learning policy object
        lp = deepcopy(self.lp)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)

        # For each row in the given contexts
        for index, row in enumerate(contexts):

            # Get random generator
            lp.rng = np.random.RandomState(seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = cdist(self.contexts, row_2d, metric=self.metric).reshape(-1)

            # Find the neighbor indices within the radius
            # np.where with a condition returns a tuple where the first element is an array of indices
            indices = np.where(distances_to_row <= self.radius)

            # If neighbors exist
            if indices[0].size > 0:

                # Fit the decisions and rewards of the neighbors
                lp.fit(self.decisions[indices], self.rewards[indices], self.contexts[indices])

                # Predict based on the neighbors
                if is_predict:
                    predictions[index] = lp.predict(row_2d)
                else:
                    predictions[index] = lp.predict_expectations(row_2d)

            else:  # When there are no neighbors
                # Random arm (or nan expectations)
                if is_predict:
                    predictions[index] = self.arms[lp.rng.randint(0, len(self.arms))]
                else:
                    predictions[index] = self.arm_to_expectation.copy()

        # Return the list of predictions
        return predictions


class _KNearest(_Neighbors):

    def __init__(self, rng: np.random.RandomState, arms: List[Arm], n_jobs: int,
                 lp: Union[_EpsilonGreedy, _ThompsonSampling, _UCB1, _Softmax, _Linear],
                 k: int, metric: str):
        super().__init__(rng, arms, n_jobs, lp, metric)

        self.k = k

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Copy Learning Policy object and set random state
        lp = deepcopy(self.lp)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)

        # For each row in the given contexts
        for index, row in enumerate(contexts):

            # Get random generator
            lp.rng = np.random.RandomState(seed=seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = cdist(self.contexts, row_2d, metric=self.metric).reshape(-1)

            # Find the k nearest neighbor indices
            indices = np.argpartition(distances_to_row, self.k - 1)[:self.k]

            # Fit the decisions and rewards of the neighbors learning from the contexts
            lp.fit(self.decisions[indices], self.rewards[indices], self.contexts[indices])

            # Predict (or predict_expectations) based on the neighbors
            # The row is used only for parametric learning policies, and it has to be 2D
            if is_predict:
                predictions[index] = lp.predict(row_2d)
            else:
                predictions[index] = lp.predict_expectations(row_2d)

        # Return the list of predictions
        return predictions
