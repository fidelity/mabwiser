# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import Dict, Callable, List, NoReturn, Optional, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import argmax, reset, Arm, Num, _BaseRNG


class _Softmax(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 tau: Optional[Union[int, float]] = 1):

        super().__init__(rng, arms, n_jobs, backend)
        self.tau = tau

        self.arm_to_sum = dict.fromkeys(self.arms, 0)
        self.arm_to_count = dict.fromkeys(self.arms, 0)
        self.arm_to_mean = dict.fromkeys(self.arms, 0)
        self.arm_to_exponent = dict.fromkeys(self.arms, 0)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the sum, count, and expectations to zero
        reset(self.arm_to_sum, 0)
        reset(self.arm_to_count, 0)
        reset(self.arm_to_mean, 0)

        # Reset warm started arms
        self._reset_arm_to_status()

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=False)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=True)

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[Arm, List[Arm]]:

        # Return the arm with maximum expectation
        expectations = self.predict_expectations(contexts)
        if isinstance(expectations, dict):
            return argmax(expectations)
        else:
            return [argmax(exp) for exp in expectations]

    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Union[Dict[Arm, Num],
                                                                                   List[Dict[Arm, Num]]]:

        # Return a random value between 0 and 1 for each arm that is "proportional" to the
        # expectation of the arm and sums to 1 by sampling from a Dirichlet distribution.
        # The Dirichlet distribution can be seen as a multivariate generalization of the Beta distribution.
        # Add a very small epsilon to ensure each of the expectations is positive.
        size = 1 if contexts is None else len(contexts)
        alpha = [v + np.finfo(float).eps for v in self.arm_to_expectation.values()]
        dirichlet_random_values = self.rng.dirichlet(alpha, size)
        expectations = [dict(zip(self.arm_to_expectation.keys(), exp)).copy() for exp in dirichlet_random_values]
        if size == 1:
            return expectations[0]
        else:
            return expectations

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        self._warm_start(arm_to_features, distance_quantile)

    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_sum[cold_arm] = deepcopy(self.arm_to_sum[warm_arm])
            self.arm_to_count[cold_arm] = deepcopy(self.arm_to_count[warm_arm])
            self.arm_to_mean[cold_arm] = deepcopy(self.arm_to_mean[warm_arm])
        self._expectation_operation()

    def _expectation_operation(self):

        # Scaling range
        max_mean = max(self.arm_to_mean.values())

        # Scale the means and calculate the natural exponents --decrement max to avoid overflow from np.exp(x)
        # Reference: https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        for arm in self.arm_to_exponent:
            self.arm_to_exponent[arm] = math.exp((self.arm_to_mean[arm] - max_mean) / self.tau)

        # Total exponent sum
        total_exponent = sum(self.arm_to_exponent.values())

        # Expectation as the ratio over total exponent
        for arm in self.arm_to_expectation:
            self.arm_to_expectation[arm] = self.arm_to_exponent[arm] / total_exponent

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        arm_rewards = rewards[decisions == arm]
        if arm_rewards.size:
            self.arm_to_sum[arm] += arm_rewards.sum()
            self.arm_to_count[arm] += arm_rewards.size
            self.arm_to_mean[arm] = self.arm_to_sum[arm] / self.arm_to_count[arm]

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.arm_to_sum[arm] = 0
        self.arm_to_count[arm] = 0
        self.arm_to_mean[arm] = 0
        self.arm_to_exponent[arm] = 0

        # Recalculate the expected values
        self._expectation_operation()

    def _drop_existing_arm(self, arm: Arm):
        self.arm_to_sum.pop(arm)
        self.arm_to_count.pop(arm)
        self.arm_to_mean.pop(arm)
        self.arm_to_exponent.pop(arm)

        # Recalculate the expected values
        self._expectation_operation()
