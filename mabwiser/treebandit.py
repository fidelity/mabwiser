# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Union, Dict, List, NoReturn, Optional, Callable

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import argmax, Arm, Num, _BaseRNG


class _TreeBandit(BaseMAB):
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 tree_parameters: Dict):
        super().__init__(rng, arms, n_jobs, backend)
        self.lp = lp
        self.tree_parameters = tree_parameters
        self.tree_parameters["random_state"] = rng.seed

        # Reset the decision tree and rewards of each arm
        self.arm_to_tree = {arm: DecisionTreeRegressor(**self.tree_parameters) for arm in self.arms}
        self.arm_to_leaf_to_rewards = {arm: defaultdict(partial(np.ndarray, 0)) for arm in self.arms}

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the decision tree and rewards of each arm
        self.arm_to_tree = {arm: DecisionTreeRegressor(**self.tree_parameters) for arm in self.arms}
        self.arm_to_leaf_to_rewards = {arm: defaultdict(partial(np.ndarray, 0)) for arm in self.arms}

        # If TS and a binarizer function is given, binarize the rewards
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.lp.is_contextual_binarized = False
            rewards = self.lp._get_binary_rewards(decisions, rewards)
            self.lp.is_contextual_binarized = True

        # Calculate fit
        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # If TS and a binarizer function is given, binarize the rewards
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.lp.is_contextual_binarized = False
            rewards = self.lp._get_binary_rewards(decisions, rewards)
            self.lp.is_contextual_binarized = True

        # Calculate fit
        self._parallel_fit(decisions, rewards, contexts)

    def predict(self, contexts: np.ndarray = None) -> Arm:

        return self._parallel_predict(contexts, is_predict=True)

    def predict_expectations(self, contexts: np.ndarray = None) -> Dict[Arm, Num]:

        return self._parallel_predict(contexts, is_predict=False)

    def _fit_arm(self, arm: Arm, decisions: np.ndarray, rewards: np.ndarray, contexts: Optional[np.ndarray] = None):

        # Create dataset for the given arm
        arm_contexts = contexts[decisions == arm]
        arm_rewards = rewards[decisions == arm]

        # Check that the dataset for the given arm is not empty
        if arm_contexts.size != 0:

            # If the arm is unfitted, train decision tree on arm dataset
            if len(self.arm_to_leaf_to_rewards[arm]) == 0:
                self.arm_to_tree[arm].fit(arm_contexts, arm_rewards)

            # For each leaf, keep a list of rewards
            # DecisionTreeClassifier's apply() method returns the indices of the nodes in the tree
            # that the specified contexts lead to. Therefore, the indices returned are not necessarily
            # consecutive/follow numerical order, but are always representative of leaf nodes.
            leaf_indices = self.arm_to_tree[arm].apply(arm_contexts)

            # Use set() to create a list of unique indices
            # These indices represent the leaves reached via the given contexts
            unique_leaf_indices = set(leaf_indices)

            for index in unique_leaf_indices:
                # Get rewards list for each leaf
                rewards_to_add = arm_rewards[leaf_indices == index]

                # Add rewards
                # NB: No need to check if index key in arm_to_rewards dict
                # thanks to defaultdict() construction
                self.arm_to_leaf_to_rewards[arm][index] = np.append(self.arm_to_leaf_to_rewards[arm][index],
                                                                    rewards_to_add)

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Get local copy of arm_to_tree, arm_to_expectation, arm_to_rewards, and arms
        # to minimize communication overhead between arms (processes) using shared objects
        arm_to_tree = deepcopy(self.arm_to_tree)
        arm_to_rewards = deepcopy(self.arm_to_leaf_to_rewards)
        arm_to_expectation = deepcopy(self.arm_to_expectation)
        arms = deepcopy(self.arms)

        # Create an empty list of predictions
        predictions = [None] * len(contexts)
        for index, row in enumerate(contexts):
            # Each row needs a separately seeded rng for reproducibility in parallel
            rng = np.random.RandomState(seed=seeds[index])

            for arm in arms:

                # Copy the row rng to the deep copied model in arm_to_tree
                arm_to_tree[arm].rng = rng

                # If there was prior data for this arm, get expectation for arm
                if arm_to_rewards[arm]:
                    # Get leaf index for that context
                    leaf_index = arm_to_tree[arm].apply([row])[0]

                    # Get the rewards list of that leaf
                    leaf_rewards = arm_to_rewards[arm][leaf_index]

                    # Create leaf lp
                    leaf_lp = self._create_leaf_lp(arm)

                    # Leaf LP: fit the same arm decision with the leaf rewards
                    leaf_lp.fit(np.asarray([arm] * len(leaf_rewards)), leaf_rewards)

                    # Leaf LP: predict expectation
                    arm_to_expectation[arm] = leaf_lp.predict_expectations()[arm]

            if is_predict:
                # Return a random arm with less than epsilon probability
                if isinstance(self.lp, _EpsilonGreedy) and self.rng.rand() < self.lp.epsilon:
                    predictions[index] = self.arms[self.rng.randint(0, len(self.arms))]
                else:
                    predictions[index] = argmax(arm_to_expectation)
            else:
                predictions[index] = arm_to_expectation.copy()

        # Return list of predictions
        return predictions

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        self.lp.add_arm(arm, binarizer)
        self.arm_to_tree[arm] = DecisionTreeRegressor(**self.tree_parameters)
        self.arm_to_leaf_to_rewards[arm] = defaultdict(partial(np.ndarray, 0))

    def _create_leaf_lp(self, arm: Arm):

        # Create a new learning policy object for each leaf
        # This avoids sharing the same object between different arms and leaves.
        if isinstance(self.lp, _EpsilonGreedy):
            leaf_lp = _EpsilonGreedy(self.rng, [arm], self.n_jobs, self.backend, self.lp.epsilon)
        elif isinstance(self.lp, _ThompsonSampling):
            leaf_lp = _ThompsonSampling(self.rng, [arm], self.n_jobs, self.backend, self.lp.binarizer)
        elif isinstance(self.lp, _UCB1):
            leaf_lp = _UCB1(self.rng, [arm], self.n_jobs, self.backend, self.lp.alpha)
        else:
            raise ValueError("Incompatible leaf lp for TreeBandit: ", self.lp)

        return leaf_lp
