# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, NoReturn
import numpy as np

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.utils import Arm, reset, _BaseRNG


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

        # Select an arm randomized by expectation probability
        # TODO: this would not work for negative rewards!
        return self.rng.choice(self.arms, p=list(self.arm_to_expectation.values()))

    def _normalize_expectations(self):
        # TODO: this would not work for negative rewards!
        total = sum(self.arm_to_expectation.values())
        if total == 0:
            # set equal probabilities
            reset(self.arm_to_expectation, 1.0 / len(self.arms))
        else:
            for k, v in self.arm_to_expectation.items():
                self.arm_to_expectation[k] = v / total
