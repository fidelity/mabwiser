# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides underlying Mab classes for the simulator
"""

import abc

from collections import defaultdict
from copy import deepcopy
from itertools import chain

from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1

from mabwiser.configs.constants import DistanceMetrics


from typing import List, Optional, Union

import numpy as np
import pandas as pd

from mabwiser._version import __author__, __copyright__, __email__, __version__

from mabwiser.neighbors.fixed import _Neighbors

from mabwiser.configs.calls import LPCall

from mabwiser.utilities.general import get_stats, get_context_hash

from mabwiser.utilities.random import _BaseRNG, create_rng

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


class _NeighborsSimulator(_Neighbors, abc.ABC):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        lp: Union[
            _EpsilonGreedy,
            _Linear,
            _Popularity,
            _Random,
            _Softmax,
            _ThompsonSampling,
            _UCB1,
        ],
        metric: DistanceMetrics,
        is_quick: bool,
        backend: Optional[str] = None,
        no_nhood_prob_of_arm: Optional[List] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric=metric, no_nhood_prob_of_arm=no_nhood_prob_of_arm)

        self.is_quick = is_quick
        self.neighborhood_arm_to_stat = []
        self.raw_rewards = None
        self.row_arm_to_expectation = []
        self.distances = None
        self.is_contextual = True
        self.neighborhood_sizes = []

    def fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None
    ):
        if LPCall.isinstance(self.lp, "ThompsonSampling") and self.lp.binarizer:
            self.raw_rewards = rewards.copy()

        super().fit(decisions, rewards, contexts)

    def partial_fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None
    ):
        if LPCall.isinstance(self.lp, "ThompsonSampling") and self.lp.binarizer:
            self.raw_rewards = np.concatenate((self.raw_rewards, rewards.copy()))

        super().partial_fit(decisions, rewards, contexts)

    def predict(self, contexts: Optional[np.ndarray] = None):
        return self._predict_operation(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None):
        return self._predict_operation(contexts, is_predict=False)

    def calculate_distances(self, contexts: np.ndarray):

        # Partition contexts by job
        n_jobs, n_contexts, starts = self._partition_contexts(len(contexts))

        # Calculate distances in parallel
        distances = Parallel(n_jobs=n_jobs, backend=self.backend)(
            delayed(self._calculate_distances_of_batch)(
                contexts[starts[i] : starts[i + 1]]
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.distances = list(chain.from_iterable(t for t in distances))

        return self.distances

    def set_distances(self, distances):
        self.distances = distances

    def _calculate_distances_of_batch(self, contexts: np.ndarray):
        distances = [None] * len(contexts)
        for index, row in enumerate(contexts):
            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances[index] = cdist(self.contexts, row_2d, metric=self.metric).reshape(
                -1
            )
        return distances

    def _predict_operation(self, contexts, is_predict):
        # Return predict within the neighborhood
        out = self._parallel_predict(contexts, is_predict=is_predict)

        if isinstance(out[0], list):
            df = pd.DataFrame(
                out, columns=["prediction", "expectations", "size", "stats"]
            )

            if is_predict:
                self.row_arm_to_expectation = (
                    self.row_arm_to_expectation + df["expectations"].tolist()
                )
            else:
                self.row_arm_to_expectation = (
                    self.row_arm_to_expectation + df["prediction"].tolist()
                )
            if not self.is_quick:
                self.neighborhood_sizes = self.neighborhood_sizes + df["size"].tolist()
                self.neighborhood_arm_to_stat = (
                    self.neighborhood_arm_to_stat + df["stats"].tolist()
                )

            return df["prediction"].tolist()

        # Single row prediction
        else:
            prediction, expectation, size, stats = out
            if is_predict:
                self.row_arm_to_expectation = self.row_arm_to_expectation + [
                    expectation
                ]
            else:
                self.row_arm_to_expectation = self.row_arm_to_expectation + [prediction]
            if not self.is_quick:
                self.neighborhood_sizes = self.neighborhood_sizes + [size]
                self.neighborhood_arm_to_stat = self.neighborhood_arm_to_stat + [stats]
            return prediction

    def _get_nhood_predictions(self, lp, row_2d, indices, is_predict):

        nn_decisions = self.decisions[indices]
        nn_rewards = self.rewards[indices]

        if isinstance(lp, _ThompsonSampling) and self.lp.binarizer:
            nn_raw_rewards = self.raw_rewards[indices]

        arm_to_stat = {}
        if not self.is_quick:
            for arm in self.arms:
                if isinstance(lp, _ThompsonSampling) and self.lp.binarizer:
                    arm_rewards = nn_raw_rewards[nn_decisions == arm]
                else:
                    arm_rewards = nn_rewards[nn_decisions == arm]

                if len(arm_rewards > 0):
                    arm_to_stat[arm] = get_stats(arm_rewards)
                else:
                    arm_to_stat[arm] = {}

        # Fit the decisions and rewards of the neighbors
        lp.fit(nn_decisions, nn_rewards, self.contexts[indices])

        # Predict based on the neighbors
        if is_predict:
            prediction = lp.predict(row_2d)
            if isinstance(lp, _ThompsonSampling):
                arm_to_expectation = lp.arm_to_expectation.copy()
            else:
                arm_to_expectation = lp.predict_expectations(row_2d)

            return prediction, arm_to_expectation, arm_to_stat
        else:
            prediction = lp.predict_expectations(row_2d)

            return prediction, {}, arm_to_stat


class _RadiusSimulator(_NeighborsSimulator):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        lp: Union[
            _EpsilonGreedy,
            _Linear,
            _Popularity,
            _Random,
            _Softmax,
            _ThompsonSampling,
            _UCB1,
        ],
        radius: float,
        metric: DistanceMetrics,
        is_quick: bool,
        backend: Optional[str] = None,
        no_nhood_prob_of_arm: Optional[List] = None,
    ):
        super().__init__(
            rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric=metric, is_quick=is_quick, no_nhood_prob_of_arm=no_nhood_prob_of_arm
        )
        self.radius = radius

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:

        # Copy learning policy object
        lp = deepcopy(self.lp)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)

        # For each row in the given contexts
        for index, row in enumerate(contexts):

            # Get random generator
            lp.rng = create_rng(seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = self.distances[start_index + index]

            # Find the neighbor indices within the radius
            # np.where with a condition returns a tuple where the first element is an array of indices
            indices = np.where(distances_to_row <= self.radius)

            # If neighbors exist
            if indices[0].size > 0:

                prediction, exp, stats = self._get_nhood_predictions(
                    lp, row_2d, indices, is_predict
                )
                predictions[index] = [prediction, exp, len(indices[0]), stats]

            else:  # When there are no neighbors

                # Random arm (or nan expectations)
                prediction = self._get_no_nhood_predictions(lp, is_predict)
                predictions[index] = [prediction, {}, 0, {}]

        # Return the list of predictions
        return predictions


class _KNearestSimulator(_NeighborsSimulator):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        lp: Union[
            _EpsilonGreedy,
            _Linear,
            _Popularity,
            _Random,
            _Softmax,
            _ThompsonSampling,
            _UCB1,
        ],
        k: int,
        metric: DistanceMetrics,
        is_quick: bool,
        backend: Optional[str] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric=metric, is_quick=is_quick)
        self.k = k

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:

        # Copy Learning Policy object and set random state
        lp = deepcopy(self.lp)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)

        # For each row in the given contexts
        for index, row in enumerate(contexts):

            # Get random generator
            lp.rng = create_rng(seed=seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = self.distances[start_index + index]

            # Find the k nearest neighbor indices
            indices = np.argpartition(distances_to_row, self.k - 1)[: self.k]

            prediction, exp, stats = self._get_nhood_predictions(
                lp, row_2d, indices, is_predict
            )
            predictions[index] = [prediction, exp, self.k, stats]

        # Return the list of predictions
        return predictions


class _ApproximateSimulator(_NeighborsSimulator, metaclass=abc.ABCMeta):
    def fit(
        self, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None,
    ) -> None:
        super().fit(decisions, rewards, contexts)

        # Initialize planes
        self._initialize(contexts.shape[1])

        # Fit hashes for each training context
        self._fit_operation(contexts, context_start=0)

    def partial_fit(
        self,
        decisions: np.ndarray,
        rewards: np.ndarray,
        contexts: Optional[np.ndarray] = None,
    ) -> None:
        start = len(self.contexts)

        super().partial_fit(decisions, rewards, contexts)

        # Fit hashes for each training context
        self._fit_operation(contexts, context_start=start)

    def _predict_contexts(
        self,
        contexts: np.ndarray,
        is_predict: bool,
        seeds: Optional[np.ndarray] = None,
        start_index: Optional[int] = None,
    ) -> List:
        # Copy learning policy object
        lp = deepcopy(self.lp)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)

        # For each row in the given contexts
        for index, row in enumerate(contexts):

            # Get random generator
            lp.rng = create_rng(seed=seeds[index])

            # Prepare for hashing
            row_2d = row[np.newaxis, :]
            indices = self._get_neighbors(row_2d)

            # Drop duplicates from list of neighbors
            indices = list(set(indices))

            # If neighbors exist
            if len(indices) > 0:

                prediction, exp, stats = self._get_nhood_predictions(
                    lp, row_2d, indices, is_predict
                )
                predictions[index] = [prediction, exp, len(indices), stats]

            else:  # When there are no neighbors

                # Random arm (or nan expectations)
                prediction = self._get_no_nhood_predictions(lp, is_predict)
                predictions[index] = [prediction, {}, 0, {}]

        return predictions

    @abc.abstractmethod
    def _get_neighbors(self, row_2d):
        """Abstract method to be implemented by child classes."""
        pass

    @abc.abstractmethod
    def _initialize(self, dimensions):
        """Abstract method to be implemented by child classes."""
        pass

    @abc.abstractmethod
    def _fit_operation(self, contexts, context_start):
        """Abstract method to be implemented by child classes."""
        pass


class _LSHSimulator(_ApproximateSimulator):
    def __init__(
        self,
        rng: _BaseRNG,
        arms: List[str],
        n_jobs: int,
        lp: Union[
            _EpsilonGreedy,
            _Linear,
            _Popularity,
            _Random,
            _Softmax,
            _ThompsonSampling,
            _UCB1,
        ],
        n_dimensions: int,
        n_tables: int,
        is_quick: bool,
        backend: Optional[str] = None,
        no_nhood_prob_of_arm: Optional[List] = None,
    ):
        super().__init__(
            rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric="simhash", is_quick=is_quick, no_nhood_prob_of_arm=no_nhood_prob_of_arm
        )

        # Properties for hash tables
        self.n_dimensions = n_dimensions
        self.n_tables = n_tables
        self.buckets = 2**n_dimensions

        # Initialize dictionaries for planes and hash table
        self.table_to_hash_to_index = {
            k: defaultdict(list) for k in range(self.n_tables)
        }
        self.table_to_plane = {i: [] for i in range(self.n_tables)}

    def _add_neighbors(self, hash_values, k, h, context_start):
        if context_start > 0:
            neighbors = np.where(hash_values == h)[0] + context_start
        else:
            neighbors = np.where(hash_values == h)[0]
        self.table_to_hash_to_index[k][h] += list(neighbors)

    def _fit_operation(self, contexts, context_start):
        # Get hashes for each hash table for each training context
        for k in self.table_to_plane.keys():

            n_contexts = len(contexts)

            # Partition contexts by job
            n_jobs, n_contexts, starts = self._partition_contexts(n_contexts)

            # Get hashes in parallel
            hash_values = Parallel(n_jobs=n_jobs, backend=self.backend)(
                delayed(get_context_hash)(
                    contexts[starts[i] : starts[i + 1]], self.table_to_plane[k]
                )
                for i in range(n_jobs)
            )

            # Reduce
            hash_values = list(chain.from_iterable(t for t in hash_values))

            # Get list of unique hashes - list is sparse, there should be collisions
            hash_keys = np.unique(hash_values)

            # For each hash, get the indices of contexts with that hash
            Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(self._add_neighbors)(hash_values, k, h, context_start)
                for h in hash_keys
            )

    def _initialize(self, n_rows):
        self.table_to_plane = {
            i: self.rng.standard_normal(size=(n_rows, self.n_dimensions))
            for i in self.table_to_plane.keys()
        }

    def _get_neighbors(self, row_2d):
        indices = list()

        # Get list of neighbors from each hash table based on the hash values of the new context
        for k in self.table_to_plane.keys():
            hash_value = get_context_hash(row_2d, self.table_to_plane[k])
            indices += self.table_to_hash_to_index[k][hash_value[0]]
        return indices
