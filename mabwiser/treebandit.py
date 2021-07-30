# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Union, Dict, List, NoReturn, Optional, Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import reset, argmax, Arm, Num, _BaseRNG


class _TreeBandit(BaseMAB):
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 tree_parameters: Dict):
        super().__init__(rng, arms, n_jobs, backend)
        self.lp = lp
        self.tree_parameters = tree_parameters
        self.tree_parameters["random_state"] = rng.seed

        self.arm_to_tree = None
        self.arm_to_rewards = None

        # Initialize the arm expectations to nan
        # When there are neighbors, expectations of the underlying learning policy is used
        # When there are no neighbors, return nan expectations
        reset(self.arm_to_expectation, np.nan)

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Reset the decision trees of each arm
        self.arm_to_tree = {arm: DecisionTreeClassifier(**self.tree_parameters) for arm in self.arms}
        self.arm_to_rewards = {arm: dict() for arm in self.arms}

        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.lp.is_contextual_binarized = False
            rewards = self.lp._get_binary_rewards(decisions, rewards)
            self.lp.is_contextual_binarized = True

        # Calculate fit
        self._parallel_fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> NoReturn:

        # Update rewards list at leaves
        # Keep track of arms whose trees have not been fitted, and fit them separately
        unfitted_arms = []
        for i, arm in enumerate(decisions):
            try:
                if arm not in unfitted_arms:
                    leaf_index = self.arm_to_tree[arm].apply([contexts[i]])[0]
                    self.arm_to_rewards[arm][leaf_index] = np.append(self.arm_to_rewards[arm][leaf_index], rewards[i])
            except AttributeError:
                # If arm's tree has not been fitted
                self.arm_to_tree[arm].fit(contexts[decisions == arm], rewards[decisions == arm])
                unfitted_arms.append(arm)


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

            # Fit decision tree of the arm with this dataset
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
                self.arm_to_rewards[arm][index] = arm_rewards[leaf_indices == index]

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

        # Get local copy of arm_to_tree, arm_to_expectation, arm_to_rewards, and arms
        # to minimize communication overhead between arms (processes) using shared objects
        arm_to_tree = deepcopy(self.arm_to_tree)
        arm_to_rewards = deepcopy(self.arm_to_rewards)
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
                    leaf_lp = self._get_leaf_lp(arm)

                    # Leaf LP: fit the same arm decision with the leaf rewards
                    leaf_lp.fit(np.asarray([arm] * len(leaf_rewards)), leaf_rewards)

                    # Leaf LP: predict expectation
                    arm_to_expectation[arm] = leaf_lp.predict_expectations()[arm]

            if is_predict:
                predictions[index] = argmax(arm_to_expectation)
            else:
                predictions[index] = arm_to_expectation.copy()

        # Return list of predictions
        return predictions

    def _uptake_new_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None):

        self.lp.add_arm(arm, binarizer)
        self.arm_to_tree[arm] = DecisionTreeClassifier(**self.tree_parameters)
        self.arm_to_rewards[arm] = dict()

    def _get_leaf_lp(self, arm: Arm):

        leaf_arms = [arm, -1]
        leaf_lp = None
        if isinstance(self.lp, _EpsilonGreedy):
            leaf_lp = _EpsilonGreedy(self.rng, leaf_arms, self.n_jobs, self.backend, self.lp.epsilon)
        elif isinstance(self.lp, _Popularity):
            leaf_lp = _Popularity(self.rng, leaf_arms, self.n_jobs, self.backend)
        elif isinstance(self.lp, _Random):
            leaf_lp = _Random(self.rng, leaf_arms, self.n_jobs, self.backend)
        elif isinstance(self.lp, _Softmax):
            leaf_lp = _Softmax(self.rng, leaf_arms, self.n_jobs, self.backend, self.lp.tau)
        elif isinstance(self.lp, _ThompsonSampling):
            leaf_lp = _ThompsonSampling(self.rng, leaf_arms, self.n_jobs, self.backend, self.lp.binarizer)
        elif isinstance(self.lp, _UCB1):
            leaf_lp = _UCB1(self.rng, leaf_arms, self.n_jobs, self.backend, self.lp.alpha)

        return leaf_lp
