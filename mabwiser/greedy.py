# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, NoReturn, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import reset, argmax, Arm, Num, _BaseRNG


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

        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return a random arm with less than epsilon probability
        if self.rng.rand() < self.epsilon:
            return self.arms[self.rng.randint(0, len(self.arms))]

        # Return the first arm with maximum expectation.
        return argmax(self.arm_to_expectation)

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Return a copy of expectations dictionary from arms (key) to expectations (values)
        return self.arm_to_expectation.copy()

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
