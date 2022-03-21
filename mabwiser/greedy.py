# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.configs.arm import ArmConfig
from mabwiser.utilities.general import reset, argmax
from mabwiser.utilities.random import _BaseRNG


class _EpsilonGreedy(BaseMAB):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        epsilon: float,
        backend: Optional[str] = None,
    ):
        super().__init__(rng, arms, n_jobs, backend)
        self.epsilon = epsilon

        self.arm_to_sum = dict.fromkeys(self.arms, 0)
        self.arm_to_count = dict.fromkeys(self.arms, 0)

    def fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None
    ) -> None:

        # Reset the sum, count, and expectations to zero
        reset(self.arm_to_sum, 0)
        reset(self.arm_to_count, 0)
        reset(self.arm_to_expectation, 0)

        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None
    ) -> None:
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: Optional[np.ndarray] = None) -> str:

        # Return the first arm with maximum expectation
        return argmax(self.predict_expectations())

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[str, float]:

        # Return a random expectation (between 0 and 1) for each arm with epsilon probability,
        # and the actual arm expectations otherwise
        if self.rng.rand() < self.epsilon:
            return dict((arm, self.rng.rand()) for arm in self.arms).copy()
        else:
            return self.arm_to_expectation.copy()

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_sum[cold_arm] = deepcopy(self.arm_to_sum[warm_arm])
            self.arm_to_count[cold_arm] = deepcopy(self.arm_to_count[warm_arm])
            self.arm_to_expectation[cold_arm] = deepcopy(
                self.arm_to_expectation[warm_arm]
            )

    def _fit_arm(
        self,
        arm: str,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        arm_rewards = rewards[decisions == arm]
        if arm_rewards.size:
            self.arm_to_sum[arm] += arm_rewards.sum()
            self.arm_to_count[arm] += arm_rewards.size
            self.arm_to_expectation[arm] = self.arm_to_sum[arm] / self.arm_to_count[arm]

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:
        pass

    def _uptake_new_arm(self, arm: ArmConfig) -> None:
        self.arm_to_sum[arm.arm] = 0
        self.arm_to_count[arm.arm] = 0

    def _drop_existing_arm(self, arm: str) -> None:
        self.arm_to_sum.pop(arm)
        self.arm_to_count.pop(arm)
