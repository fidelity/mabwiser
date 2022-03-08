# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, NoReturn
import numpy as np

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.utils import argmax, reset, Arm, Num, _BaseRNG


class _Popularity(_EpsilonGreedy):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str]):

        # Init the parent greedy policy with zero epsilon
        super().__init__(rng, arms, n_jobs, backend, epsilon=0.0)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Fit as usual greedy
        super().fit(decisions, rewards, contexts)

        # Make sure expectations sum up to 1 like probabilities
        self._normalize_expectations()

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Fit as usual greedy
        super().partial_fit(decisions, rewards, contexts)

        # Make sure expectations sum up to 1 like probabilities
        self._normalize_expectations()

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return the first arm with maximum expectation
        return argmax(self.predict_expectations())

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Return a random value between 0 and 1 for each arm that is "proportional" to the
        # expectation of the arm and sums to 1 by sampling from a Dirichlet distribution.
        # The Dirichlet distribution can be seen as a multivariate generalization of the Beta distribution.
        # Add a very small epsilon to ensure each of the expectations is positive.
        # TODO: this would not work for negative rewards!
        expectations = [v + np.finfo(float).eps for v in self.arm_to_expectation.values()]
        return dict(zip(self.arm_to_expectation.keys(), self.rng.dirichlet(expectations))).copy()

    def _normalize_expectations(self):
        # TODO: this would not work for negative rewards!
        total = sum(self.arm_to_expectation.values())
        if total == 0:
            # set equal probabilities
            reset(self.arm_to_expectation, 1.0 / len(self.arms))
        else:
            for k, v in self.arm_to_expectation.items():
                self.arm_to_expectation[k] = v / total

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        self.arm_to_sum.pop(arm)
        self.arm_to_count.pop(arm)
        self._normalize_expectations()
