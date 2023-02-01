# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union

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

        # Reset warm started arms
        self._reset_arm_to_status()

        # Calculate fit
        self._parallel_fit(decisions, rewards)

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=False)

        # Leave the calculation of expectations to predict methods

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # If rewards are non binary, convert them
        rewards = self._get_binary_rewards(decisions, rewards)

        # Calculate fit
        self._parallel_fit(decisions, rewards)

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

        # Expectation of each arm is a random sample from beta distribution with success and fail counters.
        # If contexts is None or has length of 1 generate single arm to expectations,
        # otherwise use vectorized functions to generate a list of arm to expectations with same length as contexts.
        size = 1 if contexts is None else len(contexts)
        arm_to_random_beta = dict()
        for arm in self.arm_to_expectation:
            arm_to_random_beta[arm] = self.rng.beta(self.arm_to_success_count[arm], self.arm_to_fail_count[arm], size)
        arm_to_expectation = [{arm: arm_to_random_beta[arm][i] for arm in self.arms} for i in range(size)]
        self.arm_to_expectation = arm_to_expectation[-1]
        if size == 1:
            return arm_to_expectation[0]
        else:
            return arm_to_expectation

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        self._warm_start(arm_to_features, distance_quantile)

    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_success_count[cold_arm] = deepcopy(self.arm_to_success_count[warm_arm])
            self.arm_to_fail_count[cold_arm] = deepcopy(self.arm_to_fail_count[warm_arm])

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

    def _drop_existing_arm(self, arm: Arm):
        self.arm_to_success_count.pop(arm)
        self.arm_to_fail_count.pop(arm)
