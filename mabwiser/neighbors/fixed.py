# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides classes that use non-approximate methods calculate neighbors
"""

from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist

from mabwiser.configs.constants import DistanceMetrics
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.neighbors.base import _Neighbors
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utilities.random import _BaseRNG, create_rng


class _Radius(_Neighbors):
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
        backend: Optional[str] = None,
        no_nhood_prob_of_arm: Optional[List] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric=metric, no_nhood_prob_of_arm=no_nhood_prob_of_arm)
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
            lp.rng = create_rng(seed=seeds[index])

            # Calculate the distances from the historical contexts
            # Row is 1D so convert it to 2D array for cdist using newaxis
            # Finally, reshape to flatten the output distances list
            row_2d = row[np.newaxis, :]
            distances_to_row = cdist(self.contexts, row_2d, metric=self.metric).reshape(
                -1
            )

            # Find the neighbor indices within the radius
            # np.where with a condition returns a tuple where the first element is an array of indices
            indices = np.where(distances_to_row <= self.radius)

            # If neighbors exist
            if indices[0].size > 0:
                predictions[index] = self._get_nhood_predictions(
                    lp, indices, row_2d, is_predict
                )
            else:  # When there are no neighbors
                predictions[index] = self._get_no_nhood_predictions(lp, is_predict)

        # Return the list of predictions
        return predictions


class _KNearest(_Neighbors):
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
        backend: Optional[str] = None,
    ):
        super().__init__(rng=rng, arms=arms, n_jobs=n_jobs, backend=backend, lp=lp, metric=metric, no_nhood_prob_of_arm=no_nhood_prob_of_arm)
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
            distances_to_row = cdist(self.contexts, row_2d, metric=self.metric).reshape(
                -1
            )

            # Find the k nearest neighbor indices
            indices = np.argpartition(distances_to_row, self.k - 1)[: self.k]

            predictions[index] = self._get_nhood_predictions(
                lp, indices, row_2d, is_predict
            )

        # Return the list of predictions
        return predictions
