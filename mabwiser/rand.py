# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, NoReturn, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.configs.arm import ArmConfig
from mabwiser.utilities.general import argmax
from mabwiser.utilities.random import _BaseRNG


class _Random(BaseMAB):
    def __init__(
        self, rng: _BaseRNG, arms: List[str], n_jobs: int, backend: Optional[str] = None
    ):
        super().__init__(rng, arms, n_jobs, backend)

    def fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None
    ) -> None:
        pass

    def partial_fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None
    ) -> None:
        pass

    def predict(self, contexts: Optional[np.ndarray] = None) -> str:

        # Return the first arm with maximum expectation
        return argmax(self.predict_expectations())

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[str, float]:

        # Return a random expectation (between 0 and 1) for each arm
        return dict((arm, self.rng.rand()) for arm in self.arms).copy()

    def warm_start(
        self, arm_to_features: Dict[str, List[float]], distance_quantile: float
    ) -> None:
        pass

    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        pass

    def _fit_arm(
        self,
        arm: str,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> None:
        pass

    def _uptake_new_arm(self, arm: ArmConfig) -> None:
        pass

    def _drop_existing_arm(self, arm: str) -> None:
        pass
