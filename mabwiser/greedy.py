# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import argmax, reset, Arm, Num, _BaseRNG


class _EpsilonGreedy(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 epsilon: Optional[float] = 0.05):
        super().__init__(rng, arms, n_jobs, backend)
        self.epsilon = epsilon

        self.arm_to_sum = dict.fromkeys(self.arms, 0)
        self.arm_to_count = dict.fromkeys(self.arms, 0)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the sum, count, and expectations to zero
        reset(self.arm_to_sum, 0)
        reset(self.arm_to_count, 0)
        reset(self.arm_to_expectation, 0)

        # Reset warm started arms
        self._reset_arm_to_status()

        self._parallel_fit(decisions, rewards, contexts)

        # Update trained arms
        self._set_arms_as_trained(decisions=decisions, is_partial=False)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        self._parallel_fit(decisions, rewards, contexts)

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

        # Return a random expectation (between 0 and 1) for each arm with epsilon probability,
        # and the actual arm expectations otherwise.
        # If contexts is None or has length of 1 generate single arm to expectations,
        # otherwise use vectorized functions to generate a list of arm to expectations with same length as contexts.
        if contexts is None or len(contexts) == 1:
            if self.rng.rand() < self.epsilon:
                return dict((arm, self.rng.rand()) for arm in self.arms).copy()
            else:
                return self.arm_to_expectation.copy()
        else:
            probability = self.rng.rand(len(contexts))
            random_values = self.rng.rand((len(contexts), len(self.arms)))
            expectations = [dict(zip(self.arms, exp)).copy() if probability[index] < self.epsilon
                            else self.arm_to_expectation.copy() for index, exp in enumerate(random_values)]
            return expectations

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        self._warm_start(arm_to_features, distance_quantile)

    def _copy_arms(self, cold_arm_to_warm_arm):
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_sum[cold_arm] = deepcopy(self.arm_to_sum[warm_arm])
            self.arm_to_count[cold_arm] = deepcopy(self.arm_to_count[warm_arm])
            self.arm_to_expectation[cold_arm] = deepcopy(self.arm_to_expectation[warm_arm])

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        arm_rewards = rewards[decisions == arm]
        if arm_rewards.size:
            self.arm_to_sum[arm] += arm_rewards.sum()
            self.arm_to_count[arm] += arm_rewards.size
            self.arm_to_expectation[arm] = self.arm_to_sum[arm] / self.arm_to_count[arm]

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        self.arm_to_sum[arm] = 0
        self.arm_to_count[arm] = 0

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        self.arm_to_sum.pop(arm)
        self.arm_to_count.pop(arm)
