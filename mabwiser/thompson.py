# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, NoReturn, Optional, Callable

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import Arm, Num, reset, argmax, _BaseRNG


class _ThompsonSampling(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 binarizer: Optional[Callable] = None):
        super().__init__(rng, arms, n_jobs, backend)
        self.binarizer = binarizer

        # Track whether the rewards have been binarized already by a context policy external
        self.is_contextual_binarized = False
        self.arm_to_success_count = dict.fromkeys(self.arms, 1)
        self.arm_to_fail_count = dict.fromkeys(self.arms, 1)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # If rewards are non binary, convert them
        rewards = self._get_binary_rewards(decisions, rewards)

        # Reset the success and failure counters to 1 (beta distribution is undefined for 0)
        reset(self.arm_to_success_count, 1)
        reset(self.arm_to_fail_count, 1)

        # Calculate fit
        self._parallel_fit(decisions, rewards)

        # Leave the calculation of expectations to predict methods

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # If rewards are non binary, convert them
        rewards = self._get_binary_rewards(decisions, rewards)

        # Calculate fit
        self._parallel_fit(decisions, rewards)

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return the arm with maximum expectation. If multiple max value exists, return the first one
        return argmax(self.predict_expectations())

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Expectation of each arm is a random sample from beta distribution with  success and fail counters
        for arm in self.arm_to_expectation:
            self.arm_to_expectation[arm] = self.rng.beta(self.arm_to_success_count[arm],
                                                         self.arm_to_fail_count[arm])

        # Return a copy of expectations dictionary from arms (key) to expectations (values)
        return self.arm_to_expectation.copy()

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        arm_rewards = rewards[decisions == arm]
        count_of_ones = arm_rewards.sum()
        self.arm_to_success_count[arm] += count_of_ones
        self.arm_to_fail_count[arm] += len(arm_rewards) - count_of_ones

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _get_binary_rewards(self, decisions: np.ndarray, rewards: np.ndarray):

        # If a binarizer function is given and binarization has not taken place already in a neighborhood policy
        if self.binarizer and not self.is_contextual_binarized:
            return np.fromiter((self.binarizer(decisions[index], value)  # convert every decision-reward pair to binary
                                for index, value in enumerate(rewards)), rewards.dtype)
        else:
            return rewards

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        # Don't override the existing binarizer unless a new one is given
        if binarizer:
            self.binarizer = binarizer
        self.arm_to_success_count[arm] = 1
        self.arm_to_fail_count[arm] = 1
