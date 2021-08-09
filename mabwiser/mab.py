#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the public interface of the **MABWiser Library** providing access to the following modules:

    - ``MAB``
    - ``LearningPolicy``
    - ``NeighborhoodPolicy``
"""

from typing import List, Union, Dict, NamedTuple, NoReturn, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier

from mabwiser._version import __author__, __email__, __version__, __copyright__
from mabwiser.approximate import _LSHNearest
from mabwiser.clusters import _Clusters
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.neighbors import _KNearest, _Radius
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.treebandit import _TreeBandit
from mabwiser.ucb import _UCB1
from mabwiser.utils import Constants, Arm, Num, check_true, check_false, create_rng

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


class LearningPolicy(NamedTuple):
    class EpsilonGreedy(NamedTuple):
        """Epsilon Greedy Learning Policy.

        This policy selects the arm with the highest expected reward with probability 1 - :math:`\\epsilon`,
        and with probability :math:`\\epsilon` it selects an arm at random for exploration.

        Attributes
        ----------
        epsilon: Num
            The probability of selecting a random arm for exploration.
            Integer or float. Must be between 0 and 1.
            Default value is 0.05.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.25), seed=123456)
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm1'
        """
        epsilon: Num = 0.05

        def _validate(self):
            check_true(isinstance(self.epsilon, (int, float)), TypeError("Epsilon must be an integer or float."))
            check_true(0 <= self.epsilon <= 1, ValueError("The value of epsilon must be between 0 and 1."))

    class LinTS(NamedTuple):
        """ LinTS Learning Policy

        For each arm LinTS trains a ridge regression and
        creates a multivariate normal distribution for the coefficients using the
        calculated coefficients as the mean and the covariance as:

        .. math::
            \\alpha^{2} (x_i^{T}x_i + \\lambda * I_d)^{-1}

        The normal distribution is randomly sampled to obtain
        expected coefficients for the ridge regression for each
        prediction.

        :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
        Higher :math:`\\alpha` values promote more exploration.

        The multivariate normal distribution uses Cholesky decomposition to guarantee deterministic behavior.
        This method requires that the covariance is a positive definite matrix.
        To ensure this is the case, alpha and l2_lambda are required to be greater than zero.

        Attributes
        ----------
        alpha: Num
            The multiplier to determine the degree of exploration.
            Integer or float. Must be greater than zero.
            Default value is 1.0.
        l2_lambda: Num
            The regularization strength.
            Integer or float. Must be greater than zero.
            Default value is 1.0.
        arm_to_scaler: Dict[Arm, Callable]
            Standardize context features by arm.
            Dictionary mapping each arm to a scaler object. It is assumed
            that the scaler objects are already fit and will only be used
            to transform context features.
            Default value is None.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.LinTS(alpha=0.25))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[3, 2, 0, 1]])
            'Arm2'
        """
        alpha: Num = 1.0
        l2_lambda: Num = 1.0
        arm_to_scaler: Dict[Arm, Callable] = None

        def _validate(self):
            check_true(isinstance(self.alpha, (int, float)), TypeError("Alpha must be an integer or float."))
            check_true(0 < self.alpha, ValueError("The value of alpha must be greater than zero."))
            check_true(isinstance(self.l2_lambda, (int, float)), TypeError("L2_lambda must be an integer or float."))
            check_true(0 < self.l2_lambda, ValueError("The value of l2_lambda must be greater than zero."))
            if self.arm_to_scaler is not None:
                check_true(isinstance(self.arm_to_scaler, dict), TypeError("Arm_to_scaler must be a dictionary"))

    class LinUCB(NamedTuple):
        """LinUCB Learning Policy.

        This policy trains a ridge regression for each arm.
        Then, given a given context, it predicts a regression value
        and calculates the upper confidence bound of that prediction.
        The arm with the highest highest upper bound is selected.

        The UCB for each arm is calculated as:

        .. math::
            UCB = x_i \\beta + \\alpha \\sqrt{(x_i^{T}x_i + \\lambda * I_d)^{-1}x_i}

        Where :math:`\\beta` is the matrix of the ridge regression coefficients, :math:`\\lambda` is the regularization
        strength, and I_d is a dxd identity matrix where d is the number of features in the context data.

        :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
        Higher :math:`\\alpha` values promote more exploration.

        Attributes
        ----------
        alpha: Num
            The parameter to control the exploration.
            Integer or float. Cannot be negative.
            Default value is 1.0.
        l2_lambda: Num
            The regularization strength.
            Integer or float. Cannot be negative.
            Default value is 1.0.
        arm_to_scaler: Dict[Arm, Callable]
            Standardize context features by arm.
            Dictionary mapping each arm to a scaler object. It is assumed
            that the scaler objects are already fit and will only be used
            to transform context features.
            Default value is None.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.LinUCB(alpha=1.25))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[3, 2, 0, 1]])
            'Arm2'
        """
        alpha: Num = 1.0
        l2_lambda: Num = 1.0
        arm_to_scaler: Dict[Arm, Callable] = None

        def _validate(self):
            check_true(isinstance(self.alpha, (int, float)), TypeError("Alpha must be an integer or float."))
            check_true(0 <= self.alpha, ValueError("The value of alpha cannot be negative."))
            check_true(isinstance(self.l2_lambda, (int, float)), TypeError("L2_lambda must be an integer or float."))
            check_true(0 <= self.l2_lambda, ValueError("The value of l2_lambda cannot be negative."))
            if self.arm_to_scaler is not None:
                check_true(isinstance(self.arm_to_scaler, dict), TypeError("Arm_to_scaler must be a dictionary"))

    class Popularity(NamedTuple):
        """Randomized Popularity Learning Policy.

        Returns a randomized popular arm for each prediction.
        The probability of selection for each arm is weighted by their mean reward.
        It assumes that the rewards are non-negative.

        The probability of selection is calculated as:

        .. math::
            P(arm) = \\frac{ \\mu_i } { \\Sigma{ \\mu }  }

        where :math:`\\mu_i` is the mean reward for that arm.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> mab = MAB(list_of_arms, LearningPolicy.Popularity())
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm1'
        """

        def _validate(self):
            pass

    class Random(NamedTuple):
        """Random Learning Policy.

        Returns a random arm for each prediction.
        The probability of selection for each arm is uniformly at random.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> mab = MAB(list_of_arms, LearningPolicy.Random())
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm2'
        """

        def _validate(self):
            pass

    class Softmax(NamedTuple):
        """Softmax Learning Policy.

        This policy selects each arm with a probability proportionate to its average reward.
        The average reward is calculated as a logistic function with each probability as:

        .. math::
            P(arm) = \\frac{ e ^  \\frac{\\mu_i - \\max{\\mu}}{ \\tau } }
            { \\Sigma{e ^  \\frac{\\mu - \\max{\\mu}}{ \\tau }}  }

        where :math:`\\mu_i` is the mean reward for that arm and :math:`\\tau` is the "temperature" to determine the degree of
        exploration.

        Attributes
        ----------
        tau: Num
             The temperature to control the exploration.
             Integer or float. Must be greater than zero.
             Default value is 1.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> mab = MAB(list_of_arms, LearningPolicy.Softmax(tau=1))
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm2'
        """
        tau: Num = 1

        def _validate(self):
            check_true(isinstance(self.tau, (int, float)), TypeError("Tau must be an integer or float."))
            check_true(0 < self.tau, ValueError("The value of tau must be greater than zero."))

    class ThompsonSampling(NamedTuple):
        """Thompson Sampling Learning Policy.

        This policy creates a beta distribution for each arm and
        then randomly samples from these distributions.
        The arm with the highest sample value is selected.

        Notice that rewards must be binary to create beta distributions.
        If rewards are not binary, see the ``binarizer`` function.

        Attributes
        ----------
        binarizer: Callable
            If rewards are not binary, a binarizer function is required.
            Given an arm decision and its corresponding reward, the binarizer function
            returns `True/False` or `0/1` to denote whether the decision counts
            as a success, i.e., `True/1` based on the reward or `False/0` otherwise.

            The function signature of the binarizer is:

            ``binarize(arm: Arm, reward: Num) -> True/False or 0/1``

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [1, 1, 1, 0]
            >>> mab = MAB(list_of_arms, LearningPolicy.ThompsonSampling())
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm2'

            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> arm_to_threshold = {'Arm1':10, 'Arm2':10}
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [10, 20, 15, 7]
            >>> def binarize(arm, reward): return reward > arm_to_threshold[arm]
            >>> mab = MAB(list_of_arms, LearningPolicy.ThompsonSampling(binarizer=binarize))
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm2'


        """
        binarizer: Callable = None

        def _validate(self):
            if self.binarizer:
                check_true(callable(self.binarizer), TypeError("Binarizer must be a callable function that "
                                                               "returns True/False or 0/1 to denote whether a given "
                                                               "reward value counts as a success for a given "
                                                               "arm decision. Specifically, the function signature is "
                                                               "binarize(arm: Arm, reward: Num) -> True/False or 0/1"))

    class UCB1(NamedTuple):
        """Upper Confidence Bound1 Learning Policy.

        This policy calculates an upper confidence bound for the mean reward of each arm.
        It greedily selects the arm with the highest upper confidence bound.

        The UCB for each arm is calculated as:

        .. math::
            UCB = \\mu_i + \\alpha \\times \\sqrt[]{\\frac{2 \\times log(N)}{n_i}}

        Where :math:`\\mu_i` is the mean for that arm,
        :math:`N` is the total number of trials, and
        :math:`n_i` is the number of times the arm has been selected.

        :math:`\\alpha` is a factor used to adjust how conservative the estimate is.
        Higher :math:`\\alpha` values promote more exploration.

        Attributes
        ----------
        alpha: Num
            The parameter to control the exploration.
            Integer of float. Cannot be negative.
            Default value is 1.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> mab = MAB(list_of_arms, LearningPolicy.UCB1(alpha=1.25))
            >>> mab.fit(decisions, rewards)
            >>> mab.predict()
            'Arm2'
        """

        alpha: Num = 1

        def _validate(self):
            check_true(isinstance(self.alpha, (int, float)), TypeError("Alpha must be an integer or float."))
            check_true(0 <= self.alpha, ValueError("The value of alpha cannot be negative."))


class NeighborhoodPolicy(NamedTuple):
    class Clusters(NamedTuple):
        """Clusters Neighborhood Policy.

        Clusters is a k-means clustering approach that uses the observations
        from the closest *cluster* with a learning policy.
        Supports ``KMeans`` and ``MiniBatchKMeans``.

        Attributes
        ----------
        n_clusters: Num
            The number of clusters. Integer. Must be at least 2. Default value is 2.
        is_minibatch: bool
            Boolean flag to use ``MiniBatchKMeans`` or not. Default value is False.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
            >>> list_of_arms = [1, 2, 3, 4]
            >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
            >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                            [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Clusters(3))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            [3, 1]
        """
        n_clusters: Num = 2
        is_minibatch: bool = False

        def _validate(self):
            check_true(isinstance(self.n_clusters, int), TypeError("The number of clusters must be an integer."))
            check_true(self.n_clusters >= 2, ValueError("The number of clusters must be at least two."))
            check_true(isinstance(self.is_minibatch, bool), TypeError("The is_minibatch flag must be a boolean."))

    class KNearest(NamedTuple):
        """KNearest Neighborhood Policy.

        KNearest is a nearest neighbors approach that selects the *k-nearest* observations
        to be used with a learning policy.

        Attributes
        ----------
        k: int
            The number of neighbors to select.
            Integer value. Must be greater than zero.
            Default value is 1.
        metric: str
            The metric used to calculate distance.
            Accepts any of the metrics supported by ``scipy.spatial.distance.cdist``.
            Default value is Euclidean distance.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
            >>> list_of_arms = [1, 2, 3, 4]
            >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
            >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                            [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                          NeighborhoodPolicy.KNearest(2, "euclidean"))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            [1, 1]
        """
        k: int = 1
        metric: str = "euclidean"

        def _validate(self):
            check_true(isinstance(self.k, int), TypeError("K must be an integer."))
            check_true((self.metric in Constants.distance_metrics),
                       ValueError("Metric must be supported by scipy.spatial.distance.cdist"))
            check_true(self.k > 0, ValueError("K must be greater than zero."))

    class LSHNearest(NamedTuple):
        """Locality-Sensitive Hashing Approximate Nearest Neighbors Policy.

        LSHNearest is a nearest neighbors approach that uses locality sensitive hashing with a simhash to
        select observations to be used with a learning policy.

        For the simhash, contexts are projected onto a hyperplane of n_context_cols x n_dimensions and each
        column of the hyperplane is evaluated for its sign, giving an ordered array of binary values.
        This is converted to a base 10 integer used as the hash code to assign the context to a hash table. This
        process is repeated for a specified number of hash tables, where each has a unique, randomly-generated
        hyperplane. To select the neighbors for a context, the hash code is calculated for each hash table and any
        contexts with the same hashes are selected as the neighbors.

        As with the radius or k value for other nearest neighbors algorithms, selecting the best number of dimensions
        and tables requires tuning. For the dimensions, a good starting point is to use the log of the square root of
        the number of rows in the training data. This will give you sqrt(n_rows) number of hashes.

        The number of dimensions and number of tables have inverse effects from each other on the number of empty
        neighborhoods and average neighborhood size. Increasing the dimensionality decreases the number of collisions,
        which increases the precision of the approximate neighborhood but also potentially increases the number of empty
        neighborhoods. Increasing the number of hash tables increases the likelihood of capturing neighbors the
        other random hyperplanes miss and increases the average neighborhood size. It should be noted that the fit
        operation is O(2**n_dimensions).

        Attributes
        ----------
        n_dimensions: int
            The number of dimensions to use for the hyperplane.
            Integer value. Must be greater than zero.
            Default value is 5.
        n_tables: int
            The number of hash tables.
            Integer value. Must be greater than zero.
            Default value is 3.
        no_nhood_prob_of_arm: None or List
            The probabilities associated with each arm. Used to select random arm if a prediction context has no neighbors.
            If not given, a uniform random distribution over all arms is assumed.
            The probabilities should sum up to 1.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
            >>> list_of_arms = [1, 2, 3, 4]
            >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
            >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                            [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                          NeighborhoodPolicy.LSHNearest(5, 3))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            [3, 1]
        """
        n_dimensions: int = 5
        n_tables: int = 3
        no_nhood_prob_of_arm: Optional[List] = None

        def _validate(self):
            check_true(isinstance(self.n_dimensions, int), TypeError("n_dimensions must be an integer."))
            check_true(self.n_dimensions > 0, ValueError("n_dimensions must be greater than zero."))
            check_true(isinstance(self.n_tables, int), TypeError("n_tables must be an integer"))
            check_true(self.n_tables > 0, ValueError("n_tables must be greater than zero."))
            check_true((self.no_nhood_prob_of_arm == None) or isinstance(self.no_nhood_prob_of_arm, List),
                       TypeError("no_nhood_prob_of_arm must be None or List."))
            if isinstance(self.no_nhood_prob_of_arm, List):
                check_true(np.isclose(sum(self.no_nhood_prob_of_arm), 1.0),
                           ValueError("no_nhood_prob_of_arm should sum up to 1.0"))

    class Radius(NamedTuple):
        """Radius Neighborhood Policy.

        Radius is a nearest neighborhood approach that selects the observations
        within a given *radius* to be used with a learning policy.

        Attributes
        ----------
        radius: Num
            The maximum distance within which to select observations.
            Integer or Float. Must be greater than zero.
            Default value is 1.
        metric: str
            The metric used to calculate distance.
            Accepts any of the metrics supported by scipy.spatial.distance.cdist.
            Default value is Euclidean distance.
        no_nhood_prob_of_arm: None or List
            The probabilities associated with each arm. Used to select random arm if a prediction context has no neighbors.
            If not given, a uniform random distribution over all arms is assumed.
            The probabilities should sum up to 1.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
            >>> list_of_arms = [1, 2, 3, 4]
            >>> decisions = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3]
            >>> rewards = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]
            >>> contexts = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],[0, 2, 2, 3, 5], [1, 3, 1, 1, 1], \
                            [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), \
                          NeighborhoodPolicy.Radius(2, "euclidean"))
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            [3, 1]
        """
        radius: Num = 0.05
        metric: str = "euclidean"
        no_nhood_prob_of_arm: Optional[List] = None

        def _validate(self):
            check_true(isinstance(self.radius, (int, float)), TypeError("Radius must be an integer or a float."))
            check_true((self.metric in Constants.distance_metrics),
                       ValueError("Metric must be supported by scipy.spatial.distance.cdist"))
            check_true(self.radius > 0, ValueError("Radius must be greater than zero."))
            check_true((self.no_nhood_prob_of_arm == None) or isinstance(self.no_nhood_prob_of_arm, List),
                       TypeError("no_nhood_prob_of_arm must be None or List."))
            if isinstance(self.no_nhood_prob_of_arm, List):
                check_true(np.isclose(sum(self.no_nhood_prob_of_arm), 1.0),
                           ValueError("no_nhood_prob_of_arm should sum up to 1.0"))

    class TreeBandit(NamedTuple):
        """TreeBandit Neighborhood Policy.

        This policy fits a decision tree for each arm using context history.
        It uses the leaves of these trees to partition the context space into regions
        and keeps a list of rewards for each leaf.
        To predict, it receives a context vector and goes to the corresponding
        leaf at each arm's tree and applies the given context-free MAB learning policy
        to predict expectations and choose an arm.

        The TreeBandit neighborhood policy is compatible with the following
        context-free learning policies only: EpsilonGreedy, ThompsonSampling and UCB1.

        The TreeBandit neighborhood policy is a modified version of
        the TreeHeuristic algorithm presented in:
        Adam N. Elmachtoub, Ryan McNellis, Sechan Oh, Marek Petrik
        A Practical Method for Solving Contextual Bandit Problems Using Decision Trees, UAI 2017

        Attributes
        ----------
        tree_parameters: Dict, **kwarg
            Parameters of the decision tree.
            The keys must match the parameters of sklearn.tree.DecisionTreeClassifier.
            When a parameter is not given, the default parameters from
            sklearn.tree.DecisionTreeClassifier will be chosen.
            Default value is an empty dictionary.

        Example
        -------
            >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
            >>> list_of_arms = ['Arm1', 'Arm2']
            >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
            >>> rewards = [20, 17, 25, 9]
            >>> contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
            >>> mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.TreeBandit())
            >>> mab.fit(decisions, rewards, contexts)
            >>> mab.predict([[3, 2, 0, 1]])
            'Arm2'

        """
        tree_parameters: Dict = {}

        def _validate(self):
            check_true(isinstance(self.tree_parameters, dict), TypeError("tree_parameters must be a dictionary."))
            tree = DecisionTreeClassifier()
            for key in self.tree_parameters.keys():
                check_true(key in tree.__dict__.keys(),
                           ValueError("sklearn.tree.DecisionTreeClassifier doesn't have a parameter " + str(key) + "."))

        def _is_compatible(self, learning_policy: LearningPolicy):
            # TreeBandit is compatible with these learning policies
            return isinstance(learning_policy, (LearningPolicy.EpsilonGreedy,
                                                LearningPolicy.UCB1,
                                                LearningPolicy.ThompsonSampling))


class MAB:
    """**MABWiser: Contextual Multi-Armed Bandit Library**

    MABWiser is a research library for fast prototyping of multi-armed bandit algorithms.
    It supports **context-free**, **parametric** and **non-parametric** **contextual** bandit models.

    Attributes
    ----------
    arms : list
        The list of all of the arms available for decisions. Arms can be integers, strings, etc.
    learning_policy : LearningPolicy
        The learning policy.
    neighborhood_policy : NeighborhoodPolicy
        The neighborhood policy.
    is_contextual : bool
        True if contextual policy is given, false otherwise. This is a read-only data field.
    seed : numbers.Rational
        The random seed to initialize the internal random number generator. This is a read-only data field.
    n_jobs: int
        This is used to specify how many concurrent processes/threads should be used for parallelized routines.
        Default value is set to 1.
        If set to -1, all CPUs are used.
        If set to -2, all CPUs but one are used, and so on.
    backend: str, optional
        Specify a parallelization backend implementation supported in the joblib library. Supported options are:
        - “loky” used by default, can induce some communication and memory overhead when exchanging input and
          output data with the worker Python processes.
        - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
        - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
          called function relies a lot on Python objects.
        Default value is None. In this case the default backend selected by joblib will be used.

    Examples
    --------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.25), seed=123456)
        >>> mab.fit(decisions, rewards)
        >>> mab.predict()
        'Arm1'
        >>> mab.add_arm('Arm3')
        >>> mab.partial_fit(['Arm3'], [30])
        >>> mab.predict()
        'Arm3'

        >>> from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1', 'Arm2']
        >>> rewards = [20, 17, 25, 9, 11]
        >>> contexts = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]]
        >>> contextual_mab = MAB(arms, LearningPolicy.EpsilonGreedy(), NeighborhoodPolicy.KNearest(k=3))
        >>> contextual_mab.fit(decisions, rewards, contexts)
        >>> contextual_mab.predict([[1, 1, 0], [1, 1, 1], [0, 1, 0]])
        ['Arm2', 'Arm2', 'Arm2']
        >>> contextual_mab.add_arm('Arm3')
        >>> contextual_mab.partial_fit(['Arm3'], [30], [[1, 1, 1]])
        >>> contextual_mab.predict([[1, 1, 1]])
        'Arm3'
    """

    def __init__(self,
                 arms: List[Arm],  # The list of arms
                 learning_policy: Union[LearningPolicy.EpsilonGreedy,
                                        LearningPolicy.Popularity,
                                        LearningPolicy.Random,
                                        LearningPolicy.Softmax,
                                        LearningPolicy.ThompsonSampling,
                                        LearningPolicy.UCB1,
                                        LearningPolicy.LinTS,
                                        LearningPolicy.LinUCB],  # The learning policy
                 neighborhood_policy: Union[None,
                                            NeighborhoodPolicy.LSHNearest,
                                            NeighborhoodPolicy.Clusters,
                                            NeighborhoodPolicy.KNearest,
                                            NeighborhoodPolicy.Radius,
                                            NeighborhoodPolicy.TreeBandit] = None,  # The context policy, optional
                 seed: int = Constants.default_seed,  # The random seed
                 n_jobs: int = 1,  # Number of parallel jobs
                 backend: str = None  # Parallel backend implementation
                 ):
        """Initializes a multi-armed bandit (MAB) with the given arguments.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        arms : List[Union[int, float, str]]
            The list of all of the arms available for decisions.
            Arms can be integers, strings, etc.
        learning_policy : LearningPolicy
            The learning policy.
        neighborhood_policy : NeighborhoodPolicy, optional
            The context policy. Default value is None.
        seed : numbers.Rational, optional
            The random seed to initialize the random number generator.
            Default value is set to Constants.default_seed.value
        n_jobs: int, optional
            This is used to specify how many concurrent processes/threads should be used for parallelized routines.
            Default value is set to 1.
            If set to -1, all CPUs are used.
            If set to -2, all CPUs but one are used, and so on.
        backend: str, optional
            Specify a parallelization backend implementation supported in the joblib library. Supported options are:
            - “loky” used by default, can induce some communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
              called function relies a lot on Python objects.
            Default value is None. In this case the default backend selected by joblib will be used.

        Raises
        ------
        TypeError:  Arms were not provided in a list.
        TypeError:  Learning policy type mismatch.
        TypeError:  Context policy type mismatch.
        TypeError:  Seed is not an integer.
        TypeError:  Number of parallel jobs is not an integer.
        TypeError:  Parallel backend is not a string.
        TypeError:  For EpsilonGreedy, epsilon must be integer or float.
        TypeError:  For LinTS, alpha must be an integer or float.
        TypeError:  For LinTS, l2_lambda must be an integer or float.
        TypeError:  For LinUCB, alpha must be an integer or float.
        TypeError:  For LinUCB, l2_lambda must be an integer or float.
        TypeError:  For Softmax, tau must be an integer or float.
        TypeError:  For ThompsonSampling, binarizer must be a callable function.
        TypeError:  For UCB, alpha must be an integer or float.
        TypeError:  For LSHNearest, n_dimensions must be an integer or float.
        TypeError:  For LSHNearest, n_tables must be an integer or float.
        TypeError:  For LSHNearest, no_nhood_prob_of_arm must be None or List that sums up to 1.0.
        TypeError:  For Clusters, n_clusters must be an integer.
        TypeError:  For Clusters, is_minibatch must be a boolean.
        TypeError:  For Radius, radius must be an integer or float.
        TypeError:  For Radius, no_nhood_prob_of_arm must be None or List that sums up to 1.0.
        TypeError:  For KNearest, k must be an integer or float.

        ValueError: Invalid number of arms.
        ValueError: Invalid values (None, NaN, Inf) in arms.
        ValueError: Duplicate values in arms.
        ValueError: Number of parallel jobs is 0.
        ValueError: For EpsilonGreedy, epsilon must be between 0 and 1.
        ValueError: For LinTS, alpha must be greater than zero.
        ValueError: For LinTS, l2_lambda must be greater than zero.
        ValueError: For LinUCB, alpha cannot be negative.
        ValueError: For LinUCB, l2_lambda cannot be negative.
        ValueError: For Softmax, tau must be greater than zero.
        ValueError: For UCB, alpha must be greater than zero.
        ValueError: For LSHNearest, n_dimensions must be gerater than zero.
        ValueError: For LSHNearest, n_tables must be gerater than zero.
        ValueError: For LSHNearest, if given, no_nhood_prob_of_arm list should sum up to 1.0.
        ValueError: For Clusters, n_clusters cannot be less than 2.
        ValueError: For Radius and KNearest, metric is not supported by scipy.spatial.distance.cdist.
        ValueError: For Radius, radius must be greater than zero.
        ValueError: For Radius, if given, no_nhood_prob_of_arm list should sum up to 1.0.
        ValueError: For KNearest, k must be greater than zero.
        """

        # Validate arguments
        MAB._validate_mab_args(arms, learning_policy, neighborhood_policy, seed, n_jobs, backend)

        # Save the arguments
        self.arms = arms.copy()
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend

        # Create the random number generator
        self._rng = create_rng(self.seed)
        self._is_initial_fit = False

        # Create the learning policy implementor
        lp = None
        if isinstance(learning_policy, LearningPolicy.EpsilonGreedy):
            lp = _EpsilonGreedy(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.epsilon)
        elif isinstance(learning_policy, LearningPolicy.Popularity):
            lp = _Popularity(self._rng, self.arms, self.n_jobs, self.backend)
        elif isinstance(learning_policy, LearningPolicy.Random):
            lp = _Random(self._rng, self.arms, self.n_jobs, self.backend)
        elif isinstance(learning_policy, LearningPolicy.Softmax):
            lp = _Softmax(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.tau)
        elif isinstance(learning_policy, LearningPolicy.ThompsonSampling):
            lp = _ThompsonSampling(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.binarizer)
        elif isinstance(learning_policy, LearningPolicy.UCB1):
            lp = _UCB1(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.alpha)
        elif isinstance(learning_policy, LearningPolicy.LinTS):
            lp = _Linear(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.l2_lambda,
                         learning_policy.alpha, "ts", learning_policy.arm_to_scaler)
        elif isinstance(learning_policy, LearningPolicy.LinUCB):
            lp = _Linear(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.l2_lambda,
                         learning_policy.alpha, "ucb", learning_policy.arm_to_scaler)
        else:
            check_true(False, ValueError("Undefined learning policy " + str(learning_policy)))

        # Create the mab implementor
        if neighborhood_policy:
            self.is_contextual = True

            # Do not use parallel fit or predict for Learning Policy when contextual
            lp.n_jobs = 1

            if isinstance(neighborhood_policy, NeighborhoodPolicy.Clusters):
                self._imp = _Clusters(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                      neighborhood_policy.n_clusters, neighborhood_policy.is_minibatch)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.LSHNearest):
                self._imp = _LSHNearest(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                        neighborhood_policy.n_dimensions, neighborhood_policy.n_tables,
                                        neighborhood_policy.no_nhood_prob_of_arm)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.KNearest):
                self._imp = _KNearest(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                      neighborhood_policy.k, neighborhood_policy.metric)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.Radius):
                self._imp = _Radius(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                    neighborhood_policy.radius, neighborhood_policy.metric,
                                    neighborhood_policy.no_nhood_prob_of_arm)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.TreeBandit):
                self._imp = _TreeBandit(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                        neighborhood_policy.tree_parameters)
            else:
                check_true(False, ValueError("Undefined context policy " + str(neighborhood_policy)))
        else:
            self.is_contextual = isinstance(learning_policy, (LearningPolicy.LinTS, LearningPolicy.LinUCB))
            self._imp = lp

    @property
    def learning_policy(self):
        """
        Creates named tuple of the learning policy based on the implementor.

        Returns
        -------
        The learning policy.

        Raises
        ------
        NotImplementedError: MAB learning_policy property not implemented for this learning policy.

        """
        if isinstance(self._imp, (_LSHNearest, _KNearest, _Radius, _TreeBandit)):
            lp = self._imp.lp
        elif isinstance(self._imp, _Clusters):
            lp = self._imp.lp_list[0]
        else:
            lp = self._imp

        if isinstance(lp, _EpsilonGreedy):
            if issubclass(type(lp), _Popularity):
                return LearningPolicy.Popularity()
            else:
                return LearningPolicy.EpsilonGreedy(lp.epsilon)
        elif isinstance(lp, _Linear):
            arm_to_scaler = dict()
            for arm in lp.arms:
                arm_to_scaler[arm] = lp.arm_to_model[arm].scaler
            if lp.regression == 'ts':
                return LearningPolicy.LinTS(lp.alpha, lp.l2_lambda, arm_to_scaler)
            else:
                return LearningPolicy.LinUCB(lp.alpha, lp.l2_lambda, arm_to_scaler)
        elif isinstance(lp, _Random):
            return LearningPolicy.Random()
        elif isinstance(lp, _Softmax):
            return LearningPolicy.Softmax(lp.tau)
        elif isinstance(lp, _ThompsonSampling):
            return LearningPolicy.ThompsonSampling(lp.binarizer)
        elif isinstance(lp, _UCB1):
            return LearningPolicy.UCB1(lp.alpha)
        else:
            raise NotImplementedError("MAB learning_policy property not implemented for this learning policy.")

    @property
    def neighborhood_policy(self):
        """
        Creates named tuple of the neighborhood policy based on the implementor.

        Returns
        -------
        The neighborhood policy
        """
        if isinstance(self._imp, _Clusters):
            return NeighborhoodPolicy.Clusters(self._imp.n_clusters, isinstance(self._imp.kmeans, MiniBatchKMeans))
        elif isinstance(self._imp, _KNearest):
            return NeighborhoodPolicy.KNearest(self._imp.k, self._imp.metric)
        elif isinstance(self._imp, _LSHNearest):
            return NeighborhoodPolicy.LSHNearest(self._imp.n_dimensions, self._imp.n_tables,
                                                 self._imp.no_nhood_prob_of_arm)
        elif isinstance(self._imp, _Radius):
            return NeighborhoodPolicy.Radius(self._imp.radius, self._imp.metric, self._imp.no_nhood_prob_of_arm)
        elif isinstance(self._imp, _TreeBandit):
            return NeighborhoodPolicy.TreeBandit(self._imp.tree_parameters)
        else:
            return None

    def add_arm(self, arm: Arm, binarizer: Callable = None, scaler: Callable = None) -> NoReturn:
        """ Adds an _arm_ to the list of arms.

        Incorporates the arm into the learning and neighborhood policies with no training data.

        Parameters
        ----------
        arm: Arm
            The new arm to be added.
        binarizer: Callable
            The new binarizer function for Thompson Sampling.
        scaler: Callable
            A scaler object from sklearn.preprocessing.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  For ThompsonSampling, binarizer must be a callable function.
        TypeError:  The standard scaler object must have a transform method.
        TypeError:  The standard scaler object must be fit with calculated ``mean_`` and ``var_`` attributes.

        ValueError: A binarizer function was provided but the learning policy is not Thompson Sampling.
        ValueError: The arm already exists.
        ValueError: The arm is ``None``.
        ValueError: The arm is ``NaN``.
        ValueError: The arm is ``Infinity``.
        """
        if binarizer:
            check_true(isinstance(self._imp, _ThompsonSampling) or isinstance(self._imp.lp, _ThompsonSampling),
                       ValueError("Learning policy must be Thompson Sampling to use a binarizer function."))

            check_true(callable(binarizer), TypeError("Binarizer must be a callable function that returns True/False "
                                                      "or 0/1 to denote whether a given reward value counts as a "
                                                      "success for a given arm decision. Specifically, the function "
                                                      "signature is binarize(arm: Arm, reward: Num) -> True/False "
                                                      "or 0/1"))

        if scaler:
            check_true(hasattr(scaler, 'transform'),
                       TypeError("Scaler must be a scaler object from sklearn.preprocessing with a transform method"))
            check_true(hasattr(scaler, 'mean_') and hasattr(scaler, 'var_'),
                       TypeError("Scaler must be fit with calculated mean_ and var_ attributes"))

        self._validate_arm(arm)
        self.arms.append(arm)
        self._imp.add_arm(arm, binarizer, scaler)

    def fit(self,
            decisions: Union[List[Arm], np.ndarray, pd.Series],  # Decisions that are made
            rewards: Union[List[Num], np.ndarray, pd.Series],  # Rewards that are received
            contexts: Union[None, List[List[Num]],
                            np.ndarray, pd.Series, pd.DataFrame] = None  # Contexts, optional
            ) -> NoReturn:
        """Fits the multi-armed bandit to the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions and rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``.
        """

        # Validate arguments
        self._validate_fit_args(decisions, rewards, contexts)

        # Convert to numpy array for efficiency
        decisions = MAB._convert_array(decisions)
        rewards = MAB._convert_array(rewards)

        # Check rewards are valid
        check_true(np.isfinite(sum(rewards)), TypeError("Rewards cannot contain None, nan or infinity."))

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts, decisions)

        # Call the fit method
        self._imp.fit(decisions, rewards, contexts)

        # Turn initial to true
        self._is_initial_fit = True

    def partial_fit(self,
                    decisions: Union[List[Arm], np.ndarray, pd.Series],
                    rewards: Union[List[Num], np.ndarray, pd.Series],
                    contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> NoReturn:
        """Updates the multi-armed bandit with the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] =
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions, rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``
        """

        # Validate arguments
        self._validate_fit_args(decisions, rewards, contexts)

        # Convert to numpy array for efficiency
        decisions = MAB._convert_array(decisions)
        rewards = MAB._convert_array(rewards)

        # Check rewards are valid
        check_true(np.isfinite(sum(rewards)), TypeError("Rewards cannot contain None, NaN or infinity."))

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts, decisions)

        # Call the fit or partial fit method
        if self._is_initial_fit:
            self._imp.partial_fit(decisions, rewards, contexts)
        else:
            self.fit(decisions, rewards, contexts)

    def predict(self,
                contexts: Union[None, List[Num], List[List[Num]],
                                np.ndarray, pd.Series, pd.DataFrame] = None  # Contexts, optional
                ) -> Union[Arm, List[Arm]]:
        """Returns the "best" arm (or arms list if multiple contexts are given) based on the expected reward.

        The definition of the *best* depends on the specified learning policy.
        Contextual learning policies and neighborhood policies require contexts data in training.
        In testing, they return the best arm given new context(s).

        Parameters
        ----------
        contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is None.
            Contexts should be ``None`` for context-free bandits and is required for contextual bandits.

        Returns
        -------
        The recommended arm or recommended arms list.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Predicting with contexts data when there is no contextual policy.
        ValueError: Contextual policy when predicting with no contexts data.
        """

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before prediction"))

        # Validate arguments
        self._validate_predict_args(contexts)

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts)

        # Return the arm with the best expectation
        return self._imp.predict(contexts)

    def predict_expectations(self,
                             contexts: Union[None, List[Num], List[List[Num]],
                                             np.ndarray, pd.Series, pd.DataFrame] = None  # Contexts, optional
                             ) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        """Returns a dictionary of arms (key) to their expected rewards (value).

        Contextual learning policies and neighborhood policies require contexts data for expected rewards.

        Parameters
        ----------
        contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context for the expected rewards. Default value is None.
            Contexts should be ``None`` for context-free bandits and is required for contextual bandits.

        Returns
        -------
        The dictionary of arms (key) to their expected rewards (value), or a list of such dictionaries.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array or pandas data frames.

        ValueError: Predicting with contexts data when there is no contextual policy.
        ValueError: Contextual policy when predicting with no contexts data.
        """

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before prediction"))

        # Validate arguments
        self._validate_predict_args(contexts)

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts)

        # Return a dictionary from arms (key) to expectations (value)
        return self._imp.predict_expectations(contexts)

    @staticmethod
    def _validate_mab_args(arms, learning_policy, neighborhood_policy, seed, n_jobs, backend) -> NoReturn:
        """
        Validates arguments for the MAB constructor.
        """

        # Arms
        check_true(isinstance(arms, list), TypeError("The arms should be provided in a list."))
        check_false(None in arms, ValueError("The arm list cannot contain None."))
        check_false(np.nan in arms, ValueError("The arm list cannot contain NaN."))
        check_false(np.Inf in arms, ValueError("The arm list cannot contain Infinity."))
        check_true(len(arms) == len(set(arms)), ValueError("The list of arms cannot contain duplicate values."))

        # Learning Policy type
        check_true(isinstance(learning_policy,
                              (LearningPolicy.EpsilonGreedy, LearningPolicy.Popularity, LearningPolicy.Random,
                               LearningPolicy.Softmax, LearningPolicy.ThompsonSampling, LearningPolicy.UCB1,
                               LearningPolicy.LinTS, LearningPolicy.LinUCB)),
                   TypeError("Learning Policy type mismatch."))

        # Learning policy value
        learning_policy._validate()

        # Contextual Policy
        if neighborhood_policy:
            check_true(isinstance(neighborhood_policy,
                                  (NeighborhoodPolicy.Clusters, NeighborhoodPolicy.KNearest,
                                   NeighborhoodPolicy.LSHNearest, NeighborhoodPolicy.Radius,
                                   NeighborhoodPolicy.TreeBandit)),
                       TypeError("Context Policy type mismatch."))
            neighborhood_policy._validate()

            # Tree-Bandit learning policy compatibility
            if isinstance(neighborhood_policy, NeighborhoodPolicy.TreeBandit):
                check_true(neighborhood_policy._is_compatible(learning_policy),
                           ValueError(
                               "Tree-Bandit is not compatible with the learning policy: " + str(learning_policy)))

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer."))

        # Parallel jobs
        check_true(isinstance(n_jobs, int), TypeError("Number of parallel jobs must be an integer."))
        check_true(n_jobs != 0, ValueError('Number of parallel jobs cannot be zero.'))
        if backend is not None:
            check_true(isinstance(backend, str), TypeError("Parallel backend must be a string."))

    def _validate_fit_args(self, decisions, rewards, contexts) -> NoReturn:
        """"
        Validates argument types for fit and partial_fit functions.
        """

        # Type check for decisions
        check_true(isinstance(decisions, (list, np.ndarray, pd.Series)),
                   TypeError("The decisions should be given as list, numpy array, or pandas series."))

        # Type check for rewards
        check_true(isinstance(rewards, (list, np.ndarray, pd.Series)),
                   TypeError("The rewards should be given as list, numpy array, or pandas series."))

        # Type check for contexts --don't use "if contexts" since it's n-dim array
        if contexts is not None:
            MAB._validate_context_type(contexts)

            # Sync contexts data with contextual policy
            check_true(self.is_contextual,
                       TypeError("Fitting contexts data requires context policy or parametric learning policy."))
            check_true((len(decisions) == len(contexts)) or (len(decisions) == 1 and isinstance(contexts, pd.Series)),
                       ValueError("Decisions and contexts should be same length: len(decision) = " +
                                  str(len(decisions)) + " vs. len(contexts) = " + str(len(contexts))))

        else:
            check_false(self.is_contextual,
                        TypeError("Fitting contextual policy or parametric learning policy requires contexts data."))

        # Length check for decisions and rewards
        check_true(len(decisions) == len(rewards), ValueError("Decisions and rewards should be same length."))

        # Thompson Sampling: works with binary rewards or requires function to convert non-binary rewards
        if isinstance(self.learning_policy, LearningPolicy.ThompsonSampling) and \
                self.learning_policy.binarizer is None:
            check_false(np.setdiff1d(rewards, [0, 0.0, 1, 1.0]).size,
                        ValueError("Thompson Sampling requires binary rewards when binarizer function is not "
                                   "provided."))

    def _validate_predict_args(self, contexts) -> NoReturn:
        """"
        Validates argument types for predict and predict_expectation functions.
        """

        # Context policy and context data should match
        if self.is_contextual:  # don't use "if contexts" since it's n-dim array
            check_true(contexts is not None, ValueError("Prediction with context policy requires context data."))
            MAB._validate_context_type(contexts)
        else:
            check_true(contexts is None, ValueError("Prediction with no context policy cannot handle context data."))

    @staticmethod
    def _validate_context_type(contexts) -> NoReturn:
        """
        Validates that context data is 2D
        """
        if isinstance(contexts, np.ndarray):
            check_true(contexts.ndim == 2,
                       TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))
        elif isinstance(contexts, list):
            check_true(np.array(contexts).ndim == 2,
                       TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))
        else:
            check_true(isinstance(contexts, (pd.Series, pd.DataFrame)),
                       TypeError("The contexts should be given as 2D list, numpy array, pandas series or data frames."))

    def _validate_arm(self, arm):
        """
        Validates new arm.
        """
        check_false(arm is None, ValueError("The arm cannot be None."))
        check_false(np.nan in [arm], ValueError("The arm cannot be NaN."))
        check_false(np.inf in [arm], ValueError("The arm cannot be Infinity."))
        check_false(arm in self.arms, ValueError("The arm is already in the list of arms."))

    @staticmethod
    def _convert_array(array_like) -> np.ndarray:
        """
        Convert given array to numpy array for efficiency.
        """
        if isinstance(array_like, np.ndarray):
            return array_like
        elif isinstance(array_like, list):
            return np.asarray(array_like)
        elif isinstance(array_like, pd.Series):
            return array_like.values
        else:
            raise NotImplementedError("Unsupported data type")

    @staticmethod
    def _convert_matrix(matrix_like, row=False) -> Union[None, np.ndarray]:
        """
        Convert contexts to numpy array for efficiency.
        For fit and partial fit, decisions must be provided.
        The numpy array need to be in C row-major order for efficiency.
        If the data is a series for a single row, set the row flag to True.
        """
        if matrix_like is None:
            return None
        elif isinstance(matrix_like, np.ndarray):
            if matrix_like.flags['C_CONTIGUOUS']:
                return matrix_like
            else:
                return np.asarray(matrix_like, order="C")
        elif isinstance(matrix_like, list):
            return np.asarray(matrix_like, order="C")
        elif isinstance(matrix_like, pd.DataFrame):
            if matrix_like.values.flags['C_CONTIGUOUS']:
                return matrix_like.values
            else:
                return np.asarray(matrix_like.values, order="C")
        elif isinstance(matrix_like, pd.Series):
            if row:
                return np.asarray(matrix_like.values, order="C").reshape(1, -1)
            else:
                return np.asarray(matrix_like.values, order="C").reshape(-1, 1)
        else:
            raise NotImplementedError("Unsupported contexts data type")

    def __convert_context(self, contexts, decisions=None) -> Union[None, np.ndarray]:
        """
        Convert contexts to numpy array for efficiency.
        For fit and partial fit, decisions must be provided.
        The numpy array need to be in C row-major order for efficiency.
        """
        if contexts is None:
            return None
        elif isinstance(contexts, np.ndarray):
            if contexts.flags['C_CONTIGUOUS']:
                return contexts
            else:
                return np.asarray(contexts, order="C")
        elif isinstance(contexts, list):
            return np.asarray(contexts, order="C")
        elif isinstance(contexts, pd.DataFrame):
            if contexts.values.flags['C_CONTIGUOUS']:
                return contexts.values
            else:
                return np.asarray(contexts.values, order="C")
        elif isinstance(contexts, pd.Series):
            # When context is a series, we need to differentiate between
            # a single context with multiple features vs. multiple contexts with single feature
            is_called_from_fit = decisions is not None

            if is_called_from_fit:
                if len(decisions) > 1:  # multiple decisions exists
                    return np.asarray(contexts.values, order="C").reshape(-1, 1)  # go from 1D to 2D
                else:  # single decision
                    return np.asarray(contexts.values, order="C").reshape(1, -1)  # go from 1D to 2D

            else:  # For predictions, compare the shape to the stored context history

                # We need to find out the number of features (to distinguish Series shape)
                if isinstance(self.learning_policy, (LearningPolicy.LinTS, LearningPolicy.LinUCB)):
                    first_arm = self.arms[0]
                    if isinstance(self._imp, _Linear):
                        num_features = self._imp.arm_to_model[first_arm].beta.size
                    else:
                        num_features = self._imp.contexts.shape[1]
                elif isinstance(self._imp, _TreeBandit):
                    # Even when fit() happened, the first arm might not necessarily have a fitted tree
                    # So we have to search for a fitted tree
                    for arm in self.arms:
                        try:
                            num_features = len(self._imp.arm_to_tree[arm].feature_importances_)
                        except:
                            continue
                else:
                    num_features = self._imp.contexts.shape[1]

                if num_features == 1:
                    return np.asarray(contexts.values, order="C").reshape(-1, 1)  # go from 1D to 2D
                else:
                    return np.asarray(contexts.values, order="C").reshape(1, -1)  # go from 1D to 2D

        else:
            raise NotImplementedError("Unsupported contexts data type")
