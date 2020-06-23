# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, Callable, List, NoReturn, Optional, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import reset, Arm, Num, _BaseRNG


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

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Generate the stopping value in the range [0, 1)
        stop = self.rng.rand()

        # For each arm, add the probability to the cumulative sum
        # until the stopping value is exceeded
        cumulative = 0
        for arm in self.arms:
            cumulative += self.arm_to_expectation[arm]
            if cumulative > stop:
                return arm

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Return a copy of expectations dictionary from arms (key) to expectations (values)
        return self.arm_to_expectation.copy()

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
