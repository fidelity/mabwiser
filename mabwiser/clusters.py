# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, Num, reset, _BaseRNG, create_rng


class _Clusters(BaseMAB):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 n_clusters: Num, is_minibatch: bool):
        super().__init__(rng, arms, n_jobs, backend)

        self.n_clusters = n_clusters

        if is_minibatch:
            self.kmeans = MiniBatchKMeans(n_clusters, random_state=rng.seed, n_init=3)
        else:
            self.kmeans = KMeans(n_clusters, random_state=rng.seed, n_init=10)

        # Create the list of learning policies for each cluster
        # Deep copy all parameters of the lp objects, except refer to the originals of rng and arms
        self.lp_list = [deepcopy(lp) for _ in range(self.n_clusters)]
        for c in range(self.n_clusters):
            self.lp_list[c].rng = rng
            self.lp_list[c].arms = arms

        self.decisions = None
        self.rewards = None
        self.contexts = None

        # Initialize the arm expectations to nan
        # When there are neighbors, expectations of the underlying learning policy is used
        # When there are no neighbors, return nan expectations
        reset(self.arm_to_expectation, np.nan)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray,
            contexts: Optional[np.ndarray] = None) -> NoReturn:

        # Set the historical data for prediction
        self.decisions = decisions
        self.contexts = contexts

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp_list[0], _ThompsonSampling) and self.lp_list[0].binarizer:
            for lp in self.lp_list:
                lp.is_contextual_binarized = False
            self.rewards = self.lp_list[0]._get_binary_rewards(decisions, rewards)
            for lp in self.lp_list:
                lp.is_contextual_binarized = True
        else:
            self.rewards = rewards

        self._fit_operation()

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray,
                    contexts: Optional[np.ndarray] = None) -> NoReturn:

        # Binarize the rewards if using Thompson Sampling
        if isinstance(self.lp_list[0], _ThompsonSampling) and self.lp_list[0].binarizer:
            for lp in self.lp_list:
                lp.is_contextual_binarized = False
            rewards = self.lp_list[0]._get_binary_rewards(decisions, rewards)
            for lp in self.lp_list:
                lp.is_contextual_binarized = True

        # Add more historical data for prediction
        self.decisions = np.concatenate((self.decisions, decisions))
        self.contexts = np.concatenate((self.contexts, contexts))
        self.rewards = np.concatenate((self.rewards, rewards))

        self._fit_operation()

    def predict(self, contexts: np.ndarray = None) -> Union[Arm, List[Arm]]:
        # Return predict within the cluster
        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        # Return predict expectations within the cluster
        return self._parallel_predict(contexts, is_predict=False)

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float):
        pass

    def _copy_arms(self, cold_arm_to_warm_arm):
        pass

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        # Update each learning policy
        for lp in self.lp_list:
            lp.add_arm(arm, binarizer)

    def _drop_existing_arm(self, arm: Arm) -> NoReturn:
        # Update each learning policy
        for lp in self.lp_list:
            lp.remove_arm(arm)

    def _fit_operation(self):

        # Train the clusters for the contexts
        self.kmeans.fit(self.contexts)
        cluster_predictions = self.kmeans.labels_

        # Train the learning policy for each cluster
        for c in range(self.n_clusters):
            indices = np.where(cluster_predictions == c)
            c_decisions = self.decisions[indices]
            c_rewards = self.rewards[indices]
            c_contexts = self.contexts[indices]
            self.lp_list[c].fit(c_decisions, c_rewards, c_contexts)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):
        pass

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Copy learning policy object
        lp_list = deepcopy(self.lp_list)

        # Identify the cluster for each context to predict
        cluster_predictions = self.kmeans.predict(contexts)

        # Obtain prediction for each context
        predictions = [None] * len(contexts)
        for index, row in enumerate(contexts):
            row_2d = row[np.newaxis, :]
            cluster = cluster_predictions[index]

            # Set random state
            lp_list[cluster].rng = create_rng(seed=seeds[index])

            # Predict based on the cluster
            if is_predict:
                predictions[index] = lp_list[cluster].predict(row_2d)
            else:
                predictions[index] = lp_list[cluster].predict_expectations(row_2d)

        # Return the list of predictions
        return predictions
