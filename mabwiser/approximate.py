# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import abc
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import List, NoReturn, Optional, Union

import numpy as np
from joblib import Parallel, delayed

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.neighbors import _Neighbors
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, _BaseRNG, create_rng


class _ApproximateNeighbors(_Neighbors, metaclass=abc.ABCMeta):

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:
        super().fit(decisions, rewards, contexts)

        # Initialize planes
        self._initialize(contexts.shape[1])

        # Fit hashes for each training context
        self._fit_operation(contexts, context_start=0)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:
        start = len(self.contexts)

        super().partial_fit(decisions, rewards, contexts)

        # Fit hashes for each training context
        self._fit_operation(contexts, context_start=start)

    @abc.abstractmethod
    def _get_neighbors(self, row_2d):
        """Abstract method to be implemented by child classes."""
        pass

    @abc.abstractmethod
    def _initialize(self, n_cols):
        """Abstract method to be implemented by child classes."""
        pass

    @abc.abstractmethod
    def _fit_operation(self, contexts, context_start):
        """Abstract method to be implemented by child classes."""
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:
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
                predictions[index] = self._get_nhood_predictions(lp, indices, row_2d, is_predict)
            else:  # When there are no neighbors
                predictions[index] = self._get_no_nhood_predictions(lp, is_predict)

        return predictions


class _LSHNearest(_ApproximateNeighbors):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 n_dimensions: int, n_tables: int, no_nhood_prob_of_arm=Optional[List]):
        super().__init__(rng, arms, n_jobs, backend, lp, metric='simhash', no_nhood_prob_of_arm=no_nhood_prob_of_arm)

        # Properties for hash tables
        self.n_dimensions = n_dimensions
        self.n_tables = n_tables
        self.buckets = 2 ** n_dimensions

        # Initialize dictionaries for planes and hash table
        self.table_to_hash_to_index = {k: defaultdict(list) for k in range(self.n_tables)}
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
                delayed(self.get_context_hash)(
                    contexts[starts[i]:starts[i + 1]],
                    self.table_to_plane[k])
                for i in range(n_jobs))

            # Reduce
            hash_values = list(chain.from_iterable(t for t in hash_values))

            # Get list of unique hashes - list is sparse, there should be collisions
            hash_keys = np.unique(hash_values)

            # For each hash, get the indices of contexts with that hash
            Parallel(n_jobs=n_jobs, require='sharedmem')(
                delayed(self._add_neighbors)(
                    hash_values, k, h, context_start)
                for h in hash_keys)

    def _initialize(self, n_cols):
        self.table_to_plane = {i: self.rng.standard_normal(size=(n_cols, self.n_dimensions))
                               for i in self.table_to_plane.keys()}

    def _get_neighbors(self, row_2d):
        indices = list()

        # Get list of neighbors from each hash table based on the hash values of the new context
        for k in self.table_to_plane.keys():
            hash_value = self.get_context_hash(row_2d, self.table_to_plane[k])
            indices += self.table_to_hash_to_index[k][hash_value[0]]

        return indices

    @staticmethod
    def get_context_hash(contexts, plane):
        # Project rows onto plane and get signs
        projection_signs = 1 * (np.dot(contexts, plane) > 0)

        # Get base 2 value of projection signs
        # Another approach is to convert to strings ('01000', '00101', '11111', etc)
        hash_values = np.zeros(contexts.shape[0])
        for i in range(plane.shape[1]):
            hash_values = hash_values + (projection_signs[:, i] * 2**i)

        return hash_values
