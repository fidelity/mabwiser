# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, NoReturn, Optional

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import Arm, Num, _BaseRNG


class _Random(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str]):
        super().__init__(rng, arms, n_jobs, backend)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        pass

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        pass

    def predict(self, contexts: np.ndarray = None) -> Arm:

        # Return a random arm
        return self.arms[self.rng.randint(0, len(self.arms))]

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        # Return a copy of expectations dictionary from arms (key) to expectations (values)
        return self.arm_to_expectation.copy()

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        pass
