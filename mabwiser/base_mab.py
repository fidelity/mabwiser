# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the abstract base class for contextual multi-armed bandit algorithms.
"""

import abc
from itertools import chain
from typing import Callable, Dict, List, NoReturn, Optional
import multiprocessing as mp

from joblib import Parallel, delayed
import numpy as np

from mabwiser.utils import Arm, Num, _BaseRNG
from mabwiser._version import __author__, __email__, __version__, __copyright__

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

        To ensure this is the case, alpha and l2_lambda are required to be greater than zero.

    Attributes
    ----------
    rng: np.random.RandomState
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
    arm_to_expectation: Dict[Arm, floot]
        The dictionary of arms (keys) to their expected rewards (values).
    """

    @abc.abstractmethod
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: str = None):
        """Abstract method.

        Creates a multi-armed bandit policy with the given arms.
        """
        self.rng: _BaseRNG = rng
        self.arms: List[Arm] = arms
        self.n_jobs: int = n_jobs
        self.backend: str = backend

        self.arm_to_expectation: Dict[Arm, float] = dict.fromkeys(self.arms, 0)

    def add_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None) -> NoReturn:
        """Introduces a new arm to the bandit.

        Adds the new arm with zero expectations and
        calls the ``_uptake_new_arm()`` function of the sub-class.
        """
        self.arm_to_expectation[arm] = 0
        self._uptake_new_arm(arm, binarizer, scaler)

    @abc.abstractmethod
    def fit(self, decisions: np.ndarray, rewards: np.ndarray,
            contexts: Optional[np.ndarray] = None) -> NoReturn:
        """Abstract method.

        Fits the multi-armed bandit to the given
        decision and reward history and corresponding contexts if any.
        """
        pass

    @abc.abstractmethod
    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:
        """Abstract method.

        Updates the multi-armed bandit with the given
        decision and reward history and corresponding contexts if any.
        """
        pass

    @abc.abstractmethod
    def predict(self, contexts: Optional[np.ndarray] = None) -> Arm:
        """Abstract method.

        Returns the predicted arm.
        """
        pass

    @abc.abstractmethod
    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Dict[Arm, Num]:
        """Abstract method.

        Returns a dictionary from arms (keys) to their expected rewards (values).
        """
        pass

    @abc.abstractmethod
    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None) -> NoReturn:
        """Abstract method.

        Updates the multi-armed bandit with the new arm.
        """
        pass

    @abc.abstractmethod
    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray,
                 contexts: Optional[np.ndarray] = None) -> NoReturn:
        """Abstract method.

        Fit operation for individual arm.
        """
        pass

    @abc.abstractmethod
    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
        """Abstract method.

        Predict operation for set of contexts.
        """
        pass

    def _parallel_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                      contexts: Optional[np.ndarray] = None):

        # Compute effective number of jobs
        n_jobs = self._effective_jobs(len(self.arms), self.n_jobs)

        # Perform parallel fit
        Parallel(n_jobs=n_jobs, require='sharedmem')(
                          delayed(self._fit_arm)(
                              arm, decisions, rewards, contexts)
                          for arm in self.arms)

    def _parallel_predict(self, contexts: np.ndarray, is_predict: bool):

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
                              contexts[starts[i]:starts[i + 1]],
                              is_predict,
                              seeds[starts[i]:starts[i + 1]],
                              starts[i])
                          for i in range(n_jobs))

        # Reduce
        predictions = list(chain.from_iterable(t for t in predictions))

        return predictions if len(predictions) > 1 else predictions[0]

    def _partition_contexts(self, n_contexts: int):

        # Compute effective number of jobs
        n_jobs = self._effective_jobs(n_contexts, self.n_jobs)

        # Partition contexts between jobs
        n_contexts_per_job = np.full(n_jobs, n_contexts // n_jobs, dtype=int)
        n_contexts_per_job[:n_contexts % n_jobs] += 1
        starts = np.cumsum(n_contexts_per_job)

        return n_jobs, n_contexts_per_job.tolist(), [0] + starts.tolist()

    @staticmethod
    def _effective_jobs(size: int, n_jobs: int):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        n_jobs = min(n_jobs, size)
        return n_jobs
