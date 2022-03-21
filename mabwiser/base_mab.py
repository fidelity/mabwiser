# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the abstract base class for contextual multi-armed bandit algorithms.
"""

import abc
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

from mabwiser._version import __author__, __copyright__, __email__, __version__
from mabwiser.configs.arm import ArmConfig
from mabwiser.utilities.distance import get_distance_threshold, get_pairwise_distances
from mabwiser.utilities.general import effective_jobs
from mabwiser.utilities.general import argmin
from mabwiser.utilities.random import _BaseRNG

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


class BaseMAB(metaclass=abc.ABCMeta):
    """Abstract base class for multi-armed bandits.

    This module is not intended to be used directly, instead it declares
    the basic skeleton of multi-armed bandits together with a set of parameters
    that are common to every bandit algorithm.

    It declares abstract methods that sub-classes can override to
    implement specific bandit policies using:

        - ``__init__`` constructor to initialize the bandit
        - ``add_arm`` method to add a new arm
        - ``fit`` method for training
        - ``partial_fit`` method for _online learning
        - ``predict_expectations`` method to retrieve the expectation of each arm
        - ``predict`` method for testing to retrieve the best arm based on the policy
        - ``remove_arm`` method for removing an arm
        - ``warm_start`` method for warm starting untrained (cold) arms

        To ensure this is the case, alpha and l2_lambda are required to be greater than zero.

    Attributes
    ----------
    rng: _BaseRNG
        The random number generator.
    arms: List
        The list of all arms.
    n_jobs: int
        This is used to specify how many concurrent processes/threads should be used for parallelized routines.
        Default value is set to 1.
        If set to -1, all CPUs are used.
        If set to -2, all CPUs but one are used, and so on.
    backend: str, optional
        Specify a parallelization backend implementation supported in the joblib library. Supported options are:
        - “loky” used by default, can induce some communication and memory overhead when exchanging input and output data with the worker Python processes.
        - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
        - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
          called function relies a lot on Python objects.
        Default value is None. In this case the default backend selected by joblib will be used.
    arm_to_expectation: Dict[Arm, float]
        The dictionary of arms (keys) to their expected rewards (values).
    cold_arm_to_warm_arm: Dict[Arm, Arm]:
        Mapping indicating what arm was used to warm-start cold arms.
    trained_arms: List[Arm]
        List of trained arms.
        Arms for which at least one decision has been observed are deemed trained.
    """

    @abc.abstractmethod
    def __init__(
        self, rng: _BaseRNG, arms: List[str], n_jobs: int, backend: Optional[str] = None
    ):
        """Abstract method.

        Creates a multi-armed bandit policy with the given arms.
        """
        self.rng: _BaseRNG = rng
        self.arms: List[str] = deepcopy(arms)
        self.n_jobs: int = n_jobs
        self.backend: str = backend

        self.arm_to_expectation: Dict[str, float] = dict.fromkeys(self.arms, 0.0)
        self.cold_arm_to_warm_arm: Dict[str, str] = dict()
        self.trained_arms: List[str] = list()

    def add_arm(self, arm: ArmConfig) -> None:
        """Introduces a new arm to the bandit.

        Adds the new arm with zero expectations and
        calls the ``_uptake_new_arm()`` function of the sub-class.
        """
        self.arms.append(arm.arm)
        self.arm_to_expectation[arm.arm] = 0
        self._uptake_new_arm(arm)

    def remove_arm(self, arm: str) -> None:
        """Removes arm from the bandit."""
        self.arms.pop(self.arms.index(arm))
        self.arm_to_expectation.pop(arm)
        self._drop_existing_arm(arm)

    @abc.abstractmethod
    def fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        """Abstract method.

        Fits the multi-armed bandit to the given
        decision and reward history and corresponding contexts if any.
        """
        pass

    @abc.abstractmethod
    def partial_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        """Abstract method.

        Updates the multi-armed bandit with the given
        decision and reward history and corresponding contexts if any.
        """
        pass

    @abc.abstractmethod
    def predict(self, contexts: Optional[np.ndarray] = None) -> str:
        """Abstract method.

        Returns the predicted arm.
        """
        pass

    @abc.abstractmethod
    def predict_expectations(
        self, contexts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Abstract method.

        Returns a dictionary from arms (keys) to their expected rewards (values).
        """
        pass

    def warm_start(
        self, arm_to_features: Dict[str, List[float]], distance_quantile: float
    ) -> None:
        self.cold_arm_to_warm_arm = self._get_cold_arm_to_warm_arm(
            arm_to_features, distance_quantile
        )
        self._copy_arms(self.cold_arm_to_warm_arm)

    @abc.abstractmethod
    def _copy_arms(self, cold_arm_to_warm_arm: Dict[str, str]) -> None:
        pass

    @abc.abstractmethod
    def _uptake_new_arm(self, arm: ArmConfig) -> None:
        """Abstract method.

        Updates the multi-armed bandit with the new arm.
        """
        pass

    @abc.abstractmethod
    def _drop_existing_arm(self, arm: str) -> None:
        """Abstract method.

        Removes existing arm from multi-armed bandit.
        """
        pass

    @abc.abstractmethod
    def _fit_arm(
        self,
        arm: str,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        """Abstract method.

        Fit operation for individual arm.
        """
        pass

    @abc.abstractmethod
    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:
        """Abstract method.

        Predict operation for set of contexts.
        """
        pass

    def _parallel_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:

        # Compute effective number of jobs
        n_jobs = effective_jobs(len(self.arms), self.n_jobs)

        # Perform parallel fit
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(self._fit_arm)(arm, decisions, rewards, contexts)
            for arm in self.arms
        )

        # Get list of arms in decisions
        # If decision is observed for cold arm, drop arm from cold arm dictionary
        arms = np.unique(decisions).tolist()
        for arm in arms:
            if arm in self.cold_arm_to_warm_arm:
                self.cold_arm_to_warm_arm.pop(arm)

        # Set/update list of arms for which at least one decision has been observed
        if len(self.trained_arms) == 0:
            self.trained_arms = arms
        else:
            self.trained_arms = np.unique(self.trained_arms + arms).tolist()

    def _parallel_predict(
        self, contexts: np.ndarray, is_predict: bool
    ) -> Union[List, str]:

        # Total number of contexts to predict
        n_contexts = len(contexts)

        # Partition contexts by job
        n_jobs, n_contexts, starts = self._partition_contexts(n_contexts)
        total_contexts = sum(n_contexts)

        # Get seed value for each context
        seeds = self.rng.randint(np.iinfo(np.int32).max, size=total_contexts)

        # Perform parallel predictions
        predictions = Parallel(n_jobs=n_jobs, backend=self.backend)(
            delayed(self._predict_contexts)(
                contexts[starts[i] : starts[i + 1]],
                is_predict,
                seeds[starts[i] : starts[i + 1]],
                starts[i],
            )
            for i in range(n_jobs)
        )

        # Reduce
        predictions = list(chain.from_iterable(t for t in predictions))

        return predictions if len(predictions) > 1 else predictions[0]

    def _partition_contexts(self, n_contexts: int) -> Tuple[int, List, List]:

        # Compute effective number of jobs
        n_jobs = effective_jobs(n_contexts, self.n_jobs)

        # Partition contexts between jobs
        n_contexts_per_job = np.full(n_jobs, n_contexts // n_jobs, dtype=int)
        n_contexts_per_job[: n_contexts % n_jobs] += 1
        starts = np.cumsum(n_contexts_per_job)

        return n_jobs, n_contexts_per_job.tolist(), [0] + starts.tolist()

    def _get_cold_arm_to_warm_arm(
        self, arm_to_features: Dict[str, List[float]], distance_quantile: float
    ) -> Dict:

        # Calculate from-to distances between all pairs of arms based on features
        # and then find minimum distance (threshold) required to warm start an untrained arm
        distance_from_to = get_pairwise_distances(arm_to_features)
        distance_threshold = get_distance_threshold(
            distance_from_to, quantile=distance_quantile
        )

        # Cold arms
        cold_arms = [arm for arm in self.arms if arm not in self.trained_arms]

        cold_arm_to_warm_arm = {}
        for cold_arm in cold_arms:

            # Collect distance from cold arm to warm arms
            arm_to_distance = {}
            for arm in self.trained_arms:
                if arm in self.arms:
                    arm_to_distance[arm] = distance_from_to[cold_arm][arm]
            if len(arm_to_distance) == 0:
                continue

            # Select the closest warm arm
            closest_arm = argmin(arm_to_distance)
            closest_distance = distance_from_to[cold_arm][closest_arm]

            # Warm start if closest distance lower than minimum required distance
            if closest_distance <= distance_threshold:
                cold_arm_to_warm_arm[cold_arm] = closest_arm

        return cold_arm_to_warm_arm
