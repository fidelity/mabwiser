# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable, Dict, List, NoReturn, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import reset, argmax, Arm, Num, _BaseRNG


class _UCB1(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 alpha: Optional[Num] = 0.05):
        super().__init__(rng, arms, n_jobs, backend)
        self.alpha = alpha

        self.total_count = 0
        self.arm_to_sum = dict.fromkeys(self.arms, 0)
        self.arm_to_count = dict.fromkeys(self.arms, 0)
        self.arm_to_mean = dict.fromkeys(self.arms, 0)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the sum, count, and expectations to zero
        reset(self.arm_to_sum, 0)
        reset(self.arm_to_count, 0)
        reset(self.arm_to_mean, 0)
        reset(self.arm_to_expectation, 0)

        # Total number of decisions
        self.total_count = len(decisions)

        # Calculate fit
        self._parallel_fit(decisions, rewards)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # Update total number of decisions
        self.total_count += len(decisions)

        # Calculate fit
        self._parallel_fit(decisions, rewards)

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return the first arm with maximum expectation
        return argmax(self.arm_to_expectation)

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Return a copy of expectations dictionary from arms (key) to expectations (values)
        return self.arm_to_expectation.copy()

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        # Fit individual arm
        arm_rewards = rewards[decisions == arm]
        if arm_rewards.size:
            self.arm_to_sum[arm] += arm_rewards.sum()
            self.arm_to_count[arm] += arm_rewards.size
            self.arm_to_mean[arm] = self.arm_to_sum[arm] / self.arm_to_count[arm]

        if self.arm_to_count[arm]:
            self.arm_to_expectation[arm] = _UCB1._get_ucb(self.arm_to_mean[arm], self.alpha,
                                                          self.total_count, self.arm_to_count[arm])

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    @staticmethod
    def _get_ucb(arm_mean, alpha, total_count, arm_count):

        # Return the mean + variance
        return arm_mean + alpha * math.sqrt((2 * math.log(total_count)) / arm_count)

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        self.arm_to_sum[arm] = 0
        self.arm_to_count[arm] = 0
        self.arm_to_mean[arm] = 0
