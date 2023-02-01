# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the abstract base class for contextual multi-armed bandit algorithms.
"""

import abc
from itertools import chain
from typing import Callable, Dict, List, NoReturn, Optional, Union
import multiprocessing as mp

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import numpy as np

from mabwiser.utils import Arm, Num, _BaseRNG, argmin
from mabwiser._version import __author__, __email__, __version__, __copyright__

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__

IS_TRAINED = 'is_trained'
IS_WARM = 'is_warm'
WARM_STARTED_BY = 'warm_started_by'


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
        - “loky” used by default, can induce some communication and memory overhead when exchanging input and output.
        - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
        - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
          called function relies a lot on Python objects.
        Default value is None. In this case the default backend selected by joblib will be used.
    arm_to_expectation: Dict[Arm, float]
        The dictionary of arms (keys) to their expected rewards (values).
    arm_to_status: Dict[Arm, dict]
        The dictionary of arms (keys) to their status (values), where the status consists of
        - ``is_trained``, which indicates whether an arm was ``fit`` or ``partial_fit``;
        - ``is_warm``, which indicates whether an arm was warm started, and therefore has a trained model associated;
        - and ``warm_started_by``, which indicates the arm that originally warm started this arm.
        Arms that were initially warm-started and then updated with ``partial_fit`` will retain ``is_warm`` as True
        with the relevant ``warm_started_by`` arm for tracking purposes.
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
        self._reset_arm_to_status()

    @property
    def trained_arms(self) -> List[Arm]:
        """List of trained arms.

        Arms for which at least one decision has been observed are deemed trained."""
        return [arm for arm in self.arms if self.arm_to_status[arm][IS_TRAINED]]

    @property
    def cold_arms(self) -> List[Arm]:
        """List of cold arms"""
        return [arm for arm in self.arms if (not self.arm_to_status[arm][IS_TRAINED] and
                                             not self.arm_to_status[arm][IS_WARM])]

    def add_arm(self, arm: Arm, binarizer: Callable = None) -> NoReturn:
        """Introduces a new arm to the bandit.

        Adds the new arm with zero expectations and
        calls the ``_uptake_new_arm()`` function of the sub-class.
        """
        self.arm_to_expectation[arm] = 0
        self._uptake_new_arm(arm, binarizer)
        self.arm_to_status[arm] = {IS_TRAINED: False, IS_WARM: False, WARM_STARTED_BY: None}

    def remove_arm(self, arm: Arm) -> NoReturn:
        """Removes arm from the bandit.
        """
        self.arm_to_expectation.pop(arm)
        self._drop_existing_arm(arm)
        self.arm_to_status.pop(arm)

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
    def predict(self, contexts: Optional[np.ndarray] = None) -> Union[Arm, List[Arm]]:
        """Abstract method.

        Returns the predicted arm.
        """
        pass

    @abc.abstractmethod
    def predict_expectations(self, contexts: Optional[np.ndarray] = None) -> Union[Dict[Arm, Num],
                                                                                   List[Dict[Arm, Num]]]:
        """Abstract method.

        Returns a dictionary from arms (keys) to their expected rewards (values).
        """
        pass

    @abc.abstractmethod
    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float) -> NoReturn:
        """Abstract method.

        Warm starts cold arms using similar warm arms based on distances between arm features.
        Only implemented for Learning Policies that make use of ``_warm_start`` method to copy arm information.
        """
        pass

    @abc.abstractmethod
    def _copy_arms(self, cold_arm_to_warm_arm: Dict[Arm, Arm]) -> NoReturn:
        pass

    @abc.abstractmethod
    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None) -> NoReturn:
        """Abstract method.

        Updates the multi-armed bandit with the new arm.
        """
        pass

    @abc.abstractmethod
    def _drop_existing_arm(self, arm: Arm):
        """Abstract method.

        Removes existing arm from multi-armed bandit.
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

    @staticmethod
    def _get_arm_distances(from_arm: Arm, arm_to_features: Dict[Arm, List[Num]], metric: str = 'cosine',
                           self_distance: int = 999999) -> Dict[Arm, Num]:
        """
        Calculates the distances of the given from_arm to all the arms.

        Distances calculated based on the feature vectors given in arm_to_features using the given distance metric.
        The distance of the arm to itself is set as the given self_distance.

        Parameters
        ---------
        from_arm: Arm
            Distances from this arm.
        arm_to_features: Dict[Arm, list[Num]]
            Features for each arm used to calculate distances.
        metric: str
            Distance metric to use.
            Default value is 'cosine'.
        self_distance: int
            The value to set as the distance to itself.
            Default value is 999999.

        Returns
        -------
        Returns distance from given arm to arm v as arm_to_distance[v].
        """

        # Find the distance of given from_arm to all arms including self
        arm_to_distance = {}
        for to_arm in arm_to_features.keys():
            if from_arm == to_arm:
                arm_to_distance[to_arm] = self_distance
            else:
                arm_to_distance[to_arm] = cdist(np.asarray([arm_to_features[from_arm]]),
                                                np.asarray([arm_to_features[to_arm]]),
                                                metric=metric)[0][0]

                # Cosine similarity can be nan when a feature vector is all-zeros
                if np.isnan(arm_to_distance[to_arm]):
                    arm_to_distance[to_arm] = self_distance

        return arm_to_distance

    @staticmethod
    def _get_pairwise_distances(arm_to_features: Dict[Arm, List[Num]], metric: str = 'cosine',
                                self_distance: int = 999999) -> Dict[Arm, Dict[Arm, Num]]:
        """
        Calculates the distances between each pair of arms.

        Distances calculated based on the feature vectors given in arm_to_features using the given distance metric.
        The distance of the arm to itself is set as the given self_distance.

        Parameters
        ---------
        arm_to_features: Dict[Arm, list[Num]]
            Features for each arm used to calculate distances.
        metric: str
            Distance metric to use.
            Default value is 'cosine'.
        self_distance: int
            The value to set as the distance to itself.
            Default value is 999999.

        Returns
        -------
        Returns the distance between two arms u and v as distance_from_to[u][v].
        """

        # For every arm, calculate its distance to all arms including itself
        distance_from_to = {}
        for from_arm in arm_to_features.keys():
            distance_from_to[from_arm] = BaseMAB._get_arm_distances(from_arm, arm_to_features, metric, self_distance)
        return distance_from_to

    @staticmethod
    def _get_distance_threshold(distance_from_to: Dict[Arm, Dict[Arm, Num]], quantile: Num,
                                self_distance: int = 999999) -> Num:
        """
        Calculates a threshold for doing warm-start conditioned on minimum pairwise distances of arms.

        Parameters
        ---------
        distance_from_to: Dict[Arm, Dict[Arm, Num]]
            Dictionary of pairwise distances from arms to arms.
        quantile: Num
            Quantile used to compute threshold.
        self_distance: int
            The distance of arm to itself.
            Default value is 999999.

        Returns
        -------
        A threshold on pairwise distance for doing warm-start.
        """

        closest_distances = []
        for arm, arm_to_distance in distance_from_to.items():

            # Get distances from one arm to others
            distances = [distance for distance in arm_to_distance.values()]

            # Get the distance to closest arm (if not equal to self_distance)
            if min(distances) != self_distance:
                closest_distances.append(min(distances))

        # Calculate threshold distance based on quantile
        threshold = np.quantile(closest_distances, q=quantile)

        return threshold

    def _get_cold_arm_to_warm_arm(self, arm_to_features, distance_quantile):

        # Calculate from-to distances between all pairs of arms based on features
        # and then find minimum distance (threshold) required to warm start an untrained arm
        distance_from_to = self._get_pairwise_distances(arm_to_features)
        distance_threshold = self._get_distance_threshold(distance_from_to, quantile=distance_quantile)

        # New cold arm to warm arm dictionary
        new_cold_arm_to_warm_arm = dict()

        for cold_arm in self.cold_arms:

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
                new_cold_arm_to_warm_arm[cold_arm] = closest_arm

        return new_cold_arm_to_warm_arm

    def _reset_arm_to_status(self):
        self.arm_to_status: Dict[Arm, dict] = {arm: {IS_TRAINED: False, IS_WARM: False,
                                                     WARM_STARTED_BY: None}
                                               for arm in self.arms}

    def _set_arms_as_trained(self, decisions: Optional[np.ndarray] = None, is_partial: bool = True):
        """Sets the given arms as trained, where arms are calculated from the ``decisions``.

        Used to update status of arms in ``fit`` and ``partial_fit`` methods. Any arm with value in decisions will
        have ``is_trained`` status updated to be True. When used in ``fit``, arms with values in decisions will also
        have warm start status re-initialized.

        """
        # Calculate arms from decisions
        arms = np.unique(decisions).tolist()

        for arm in self.arms:
            if arm in arms:
                # All system arms are now trained
                self.arm_to_status[arm][IS_TRAINED] = True

                # If fitting from scratch, arm is no longer warm started
                if not is_partial:
                    self.arm_to_status[arm][IS_WARM] = False
                    self.arm_to_status[arm][WARM_STARTED_BY] = None

    def _warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float) -> NoReturn:
        cold_arm_to_warm_arm = self._get_cold_arm_to_warm_arm(arm_to_features, distance_quantile)
        self._copy_arms(cold_arm_to_warm_arm)
        for cold_arm, warm_arm in cold_arm_to_warm_arm.items():
            self.arm_to_status[cold_arm][IS_WARM] = True
            self.arm_to_status[cold_arm][WARM_STARTED_BY] = warm_arm