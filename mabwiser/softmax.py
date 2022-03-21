# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import math
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.configs.arm import ArmConfig
from mabwiser.utilities.general import argmax, reset
from mabwiser.utilities.random import _BaseRNG


class _Softmax(BaseMAB):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        tau: float,
        backend: Optional[str] = None,
    ):

        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend)
        self.tau = tau

        self.arm_to_sum = dict.fromkeys(self.arms, 0)
        self.arm_to_count = dict.fromkeys(self.arms, 0)
        self.arm_to_mean = dict.fromkeys(self.arms, 0)
        self.arm_to_exponent = dict.fromkeys(self.arms, 0)

    def fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Reset the sum, count, and expectations to zero
        reset(self.arm_to_sum, 0)
        reset(self.arm_to_count, 0)
        reset(self.arm_to_mean, 0)

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

    def partial_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Calculate fit
        self._parallel_fit(decisions, rewards)
        self._expectation_operation()

    def predict(self, contexts: Optional[np.ndarray] = None) -> str:

        # Return the first arm with maximum expectation
        return argmax(self.predict_expectations())

    def predict_expectations(
        self, contexts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:

        # Return a random value between 0 and 1 for each arm that is "proportional" to the
        # expectation of the arm and sums to 1 by sampling from a Dirichlet distribution.
        # The Dirichlet distribution can be seen as a multivariate generalization of the Beta distribution.
        # Add a very small epsilon to ensure each of the expectations is positive.
        expectations = [
            v + np.finfo(float).eps for v in self.arm_to_expectation.values()
        ]
        return dict(
            zip(self.arm_to_expectation.keys(), self.rng.dirichlet(expectations))
        ).copy()

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_sum[cold_arm] = deepcopy(self.arm_to_sum[warm_arm])
            self.arm_to_count[cold_arm] = deepcopy(self.arm_to_count[warm_arm])
            self.arm_to_mean[cold_arm] = deepcopy(self.arm_to_mean[warm_arm])
        self._expectation_operation()

    def _expectation_operation(self) -> None:

        # Scaling range
        max_mean = max(self.arm_to_mean.values())
        # Scale the means and calculate the natural exponents -- decrement max to avoid overflow from np.exp(x)
        # Reference: https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        for arm in self.arm_to_exponent:
            self.arm_to_exponent[arm] = math.exp(
                (self.arm_to_mean[arm] - max_mean) / self.tau
            )
        # Total exponent sum
        total_exponent = sum(self.arm_to_exponent.values())

        # Expectation as the ratio over total exponent
        for arm in self.arm_to_expectation:
            self.arm_to_expectation[arm] = self.arm_to_exponent[arm] / total_exponent

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
            self.arm_to_mean[arm] = self.arm_to_sum[arm] / self.arm_to_count[arm]

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> None:
        pass

    def _uptake_new_arm(self, arm: ArmConfig) -> None:
        self.arm_to_sum[arm.arm] = 0
        self.arm_to_count[arm.arm] = 0
        self.arm_to_mean[arm.arm] = 0
        self.arm_to_exponent[arm.arm] = 0

        # Recalculate the expected values
        self._expectation_operation()

    def _drop_existing_arm(self, arm: str) -> None:
        self.arm_to_sum.pop(arm)
        self.arm_to_count.pop(arm)
        self.arm_to_mean.pop(arm)
        self.arm_to_exponent.pop(arm)

        # Recalculate the expected values
        self._expectation_operation()
