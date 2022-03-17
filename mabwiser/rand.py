# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, NoReturn, Optional, Union

import numpy as np

from mabwiser.base_mab import BaseMAB
from mabwiser.utils import argmax, Arm, Num, _BaseRNG


class _Random(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str]):
        super().__init__(rng, arms, n_jobs, backend)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        pass

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        pass

    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[Arm, List[Arm]]:

        # Return the arm with maximum expectation
        expectations = self.predict_expectations(contexts)
        if isinstance(expectations, dict):
            return argmax(expectations)
        else:
            return [argmax(exp) for exp in expectations]

    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Union[Dict[Arm, Num],
                                                                                   List[Dict[Arm, Num]]]:

        # Return a random expectation (between 0 and 1) for each arm.
        # If contexts is None or has length of 1 generate single arm to expectations,
        # otherwise use vectorized functions to generate a list of arm to expectations with same length as contexts.
        size = 1 if contexts is None else len(contexts)
        random_values = self.rng.rand((size, len(self.arms)))
        expectations = [dict(zip(self.arms, exp)).copy() for exp in random_values]
        if size == 1:
            return expectations[0]
        else:
            return expectations

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        pass

    def _copy_arms(self, cold_arm_to_warm_arm):
        pass

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):
        pass

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        pass
