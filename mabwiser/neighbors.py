# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, Num, reset, _BaseRNG, create_rng


class _Neighbors(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 metric: str, no_nhood_prob_of_arm: Optional[List] = None):
        super().__init__(rng, arms, n_jobs, backend)
        self.lp = lp
        self.metric = metric
        self.no_nhood_prob_of_arm = no_nhood_prob_of_arm

        self.decisions = None
        self.rewards = None
        self.contexts = None

        # Set warm start variables to None
        self.arm_to_features = None
        self.distance_quantile = None

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
            self.rewards = self._binarize_ts_rewards(decisions, rewards)
        else:
            self.rewards = rewards

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            rewards = self._binarize_ts_rewards(decisions, rewards)

        # Add more historical data for prediction
        self.decisions = np.concatenate((self.decisions, decisions))
        self.contexts = np.concatenate((self.contexts, contexts))
        self.rewards = np.concatenate((self.rewards, rewards))

    def predict(self, contexts: np.ndarray = None) -> Union[Arm, List[Arm]]:

        # Return predict within the neighborhood
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:

        # Return predict expectations within the neighborhood
        return self._parallel_predict(contexts, is_predict=False)

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        pass

    def _copy_arms(self, cold_arm_to_warm_arm):
        pass

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        """Abstract method to be implemented by child classes."""
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        """Abstract method to be implemented by child classes."""
        pass

    def _binarize_ts_rewards(self, decisions, rewards):
        self.lp.is_contextual_binarized = False
        rewards = self.lp._get_binary_rewards(decisions, rewards)
        self.lp.is_contextual_binarized = True

        return rewards

    def _get_nhood_predictions(self, lp, indices, row_2d, is_predict):

        # Fit the decisions and rewards of the neighbors
        lp.fit(self.decisions[indices], self.rewards[indices], self.contexts[indices])

        # Predict based on the neighbors
        if is_predict:
            return lp.predict(row_2d)
        else:
            return lp.predict_expectations(row_2d)

    def _get_no_nhood_predictions(self, lp, is_predict):

        if is_predict:
            # if no_nhood_prob_of_arm is None, select a random int
            # else, select a non-uniform random arm
            # choice returns an array, hence get zero index
            rand_int = lp.rng.choice(len(self.arms), size=1, p=self.no_nhood_prob_of_arm)[0]
            return self.arms[rand_int]
        else:
            # Expectations will be nan when there are no neighbors
            return self.arm_to_expectation.copy()

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.lp.add_arm(arm, binarizer)

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        self.lp.remove_arm(arm)


class _Radius(_Neighbors):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 radius: Num, metric: str, no_nhood_prob_of_arm=Optional[List]):
        super().__init__(rng, arms, n_jobs, backend, lp, metric, no_nhood_prob_of_arm)

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
            lp.rng = create_rng(seed=seeds[index])

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
                predictions[index] = self._get_nhood_predictions(lp, indices, row_2d, is_predict)
            else:  # When there are no neighbors
                predictions[index] = self._get_no_nhood_predictions(lp, is_predict)

        # Return the list of predictions
        return predictions


class _KNearest(_Neighbors):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 k: int, metric: str):
        super().__init__(rng, arms, n_jobs, backend, lp, metric)

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
            lp.rng = create_rng(seed=seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = cdist(self.contexts, row_2d, metric=self.metric).reshape(-1)

            # Find the k nearest neighbor indices
            indices = np.argpartition(distances_to_row, self.k - 1)[:self.k]

            predictions[index] = self._get_nhood_predictions(lp, indices, row_2d, is_predict)

        # Return the list of predictions
        return predictions
