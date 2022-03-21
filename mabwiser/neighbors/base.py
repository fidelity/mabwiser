# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides a base interface for classes that calculate neighbors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.configs.arm import ArmConfig
from mabwiser.configs.constants import DistanceMetrics
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utilities.general import reset
from mabwiser.utilities.random import _BaseRNG


class _Neighbors(BaseMAB, ABC):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        lp: Union[
            _EpsilonGreedy,
            _Linear,
            _Popularity,
            _Random,
            _Softmax,
            _ThompsonSampling,
            _UCB1,
        ],
        metric: DistanceMetrics,
        backend: Optional[str] = None,
        no_nhood_prob_of_arm: Optional[List] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend)
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

    def fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Set the historical data for prediction
        self.decisions = decisions
        self.contexts = contexts

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.rewards = self._binarize_ts_rewards(decisions, rewards)
        else:
            self.rewards = rewards

    def partial_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            rewards = self._binarize_ts_rewards(decisions, rewards)

        # Add more historical data for prediction
        self.decisions = np.concatenate((self.decisions, decisions))
        self.contexts = np.concatenate((self.contexts, contexts))
        self.rewards = np.concatenate((self.rewards, rewards))

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[List, str]:

        # Return predict within the neighborhood
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(
        self, contexts: Optional[np.ndarray] = None
    ) -> Union[List, str]:

        # Return predict expectations within the neighborhood
        return self._parallel_predict(contexts, is_predict=False)

    def warm_start(
        self, arm_to_features: Dict[str, List[float]], distance_quantile: float
    ):
        # Can only execute warm start when learning policy has been fit in _get_nhood_predictions
        self.arm_to_features = arm_to_features
        self.distance_quantile = distance_quantile

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        # Copy arms executed on learning policy in _get_nhood_predictions
        pass

    def _fit_arm(
        self,
        arm: str,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ):
        """Abstract method to be implemented by child classes."""
        pass

    @abstractmethod
    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:
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

        # Warm start
        if self.arm_to_features is not None:
            lp.warm_start(self.arm_to_features, self.distance_quantile)

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
            rand_int = lp.rng.choice(
                len(self.arms), size=1, p=self.no_nhood_prob_of_arm
            )[0]
            return self.arms[rand_int]
        else:
            # Expectations will be nan when there are no neighbors
            return self.arm_to_expectation.copy()

    def _uptake_new_arm(self, arm: ArmConfig) -> None:
        self.lp.add_arm(arm)

    def _drop_existing_arm(self, arm: str) -> None:
        self.lp.remove_arm(arm)
