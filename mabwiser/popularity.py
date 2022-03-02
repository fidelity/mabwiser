# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional
import heapq
import numpy as np

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.utils import Arm, Num, _BaseRNG


class _Popularity(_EpsilonGreedy):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 k: Optional[int] = None):

        # Init the parent greedy policy with zero epsilon
        super().__init__(rng, arms, n_jobs, backend, epsilon=0.0)
        self.k = k

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Find the k (most popular) arms with the highest arm to expectations
        if self.k is not None:
            popular_arms = heapq.nlargest(self.k, self.arm_to_expectation, key=self.arm_to_expectation.__getitem__)
        else:
            popular_arms = self.arms

        # Return a random expectation (between 0 and 1) for each of the popular arms and 0 for other arms
        return dict((arm, self.rng.rand() if arm in popular_arms else 0) for arm in self.arms).copy()
