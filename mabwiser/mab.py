#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the public interface of the **MABWiser Library** providing access to the following modules:

    - ``MAB``
    - ``LearningPolicy``
    - ``NeighborhoodPolicy``
"""

import attr
from typing import List, Union, Dict,  NoReturn, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from mabwiser._version import __author__, __email__, __version__, __copyright__
from mabwiser.approximate import _LSHNearest
from mabwiser.clusters import _Clusters
from mabwiser.configs.learning import LearningPolicy
from mabwiser.configs.neighborhood import NeighborhoodPolicy
from mabwiser.configs.validators import is_compatible
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
                                        LearningPolicy.LinGreedy,
                                        LearningPolicy.LinTS,
                                        LearningPolicy.LinUCB],  # The learning policy
                 neighborhood_policy: Optional[Union[
                                            NeighborhoodPolicy.LSHNearest,
                                            NeighborhoodPolicy.Clusters,
                                            NeighborhoodPolicy.KNearest,
                                            NeighborhoodPolicy.Radius,
                                            NeighborhoodPolicy.TreeBandit]] = None,  # The context policy, optional
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
        TypeError:  For LinGreedy, epsilon must be an integer or float.
        TypeError:  For LinGreedy, l2_lambda must be an integer or float.
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
        ValueError: For LinGreedy, epsilon must be between 0 and 1.
        ValueError: For LinGreedy, l2_lambda cannot be negative.
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
        elif isinstance(learning_policy, LearningPolicy.LinGreedy):
            lp = _Linear(self._rng, self.arms, self.n_jobs, self.backend, 0, learning_policy.epsilon,
                         learning_policy.l2_lambda, "ridge", learning_policy.scale)
        elif isinstance(learning_policy, LearningPolicy.LinTS):
            lp = _Linear(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ts", learning_policy.scale)
        elif isinstance(learning_policy, LearningPolicy.LinUCB):
            lp = _Linear(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ucb", learning_policy.scale)
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
                                        attr.asdict(neighborhood_policy))
            else:
                check_true(False, ValueError("Undefined context policy " + str(neighborhood_policy)))
        else:
            self.is_contextual = isinstance(learning_policy, (LearningPolicy.LinGreedy, LearningPolicy.LinTS,
                                                              LearningPolicy.LinUCB))
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
                return LearningPolicy.EpsilonGreedy(epsilon=lp.epsilon)
        elif isinstance(lp, _Linear):
            if lp.regression == 'ridge':
                return LearningPolicy.LinGreedy(epsilon=lp.epsilon, l2_lambda=lp.l2_lambda, scale=lp.scale)
            elif lp.regression == 'ts':
                return LearningPolicy.LinTS(alpha=lp.alpha, l2_lambda=lp.l2_lambda, scale=lp.scale)
            elif lp.regression == 'ucb':
                return LearningPolicy.LinUCB(alpha=lp.alpha, l2_lambda=lp.l2_lambda, scale=lp.scale)
            else:
                check_true(False, ValueError("Undefined regression " + str(lp.regression)))
        elif isinstance(lp, _Random):
            return LearningPolicy.Random()
        elif isinstance(lp, _Softmax):
            return LearningPolicy.Softmax(tau=lp.tau)
        elif isinstance(lp, _ThompsonSampling):
            return LearningPolicy.ThompsonSampling(binarizer=lp.binarizer)
        elif isinstance(lp, _UCB1):
            return LearningPolicy.UCB1(alpha=lp.alpha)
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
            return NeighborhoodPolicy.Clusters(
                n_clusters=self._imp.n_clusters, is_minibatch=isinstance(self._imp.kmeans, MiniBatchKMeans)
            )
        elif isinstance(self._imp, _KNearest):
            return NeighborhoodPolicy.KNearest(k=self._imp.k, metric=self._imp.metric)
        elif isinstance(self._imp, _LSHNearest):
            return NeighborhoodPolicy.LSHNearest(n_dimensions=self._imp.n_dimensions, n_tables=self._imp.n_tables,
                                                 no_nhood_prob_of_arm=self._imp.no_nhood_prob_of_arm)
        elif isinstance(self._imp, _Radius):
            return NeighborhoodPolicy.Radius(
                radius=self._imp.radius, metric=self._imp.metric, no_nhood_prob_of_arm=self._imp.no_nhood_prob_of_arm
            )
        elif isinstance(self._imp, _TreeBandit):
            return NeighborhoodPolicy.TreeBandit(tree_parameters=self._imp.tree_parameters)
        else:
            return None

    def add_arm(self, arm: Arm, binarizer: Callable = None) -> None:
        """ Adds an _arm_ to the list of arms.

        Incorporates the arm into the learning and neighborhood policies with no training data.

        Parameters
        ----------
        arm: Arm
            The new arm to be added.
        binarizer: Callable
            The new binarizer function for Thompson Sampling.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  For ThompsonSampling, binarizer must be a callable function.

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
        check_false(arm in self.arms, ValueError("The arm is already in the list of arms."))

        self._validate_arm(arm)
        self.arms.append(arm)
        self._imp.add_arm(arm, binarizer)

    def remove_arm(self, arm: Arm) -> None:
        """Removes an _arm_ from the list of arms.

        Parameters
        ----------
        arm: Arm
            The existing arm to be removed.

        Returns
        -------
        No return.

        Raises
        ------
        ValueError: The arm does not exist.
        ValueError: The arm is ``None``.
        ValueError: The arm is ``NaN``.
        ValueError: The arm is ``Infinity``.
        """

        check_true(arm in self.arms, ValueError("The arm is not in the list of arms."))

        self._validate_arm(arm)
        self.arms.remove(arm)
        self._imp.remove_arm(arm)

    def fit(self,
            decisions: Union[List[Arm], np.ndarray, pd.Series],  # Decisions that are made
            rewards: Union[List[Num], np.ndarray, pd.Series],  # Rewards that are received
            contexts: Union[None, List[List[Num]],
                            np.ndarray, pd.Series, pd.DataFrame] = None  # Contexts, optional
            ) -> None:
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
                    contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> None:
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
        contexts : Union[None, List[Num], List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context for the expected rewards. Default value is None.
            If contexts is not ``None`` for context-free bandits, the predictions returned will be a
            list of the same length as contexts.

        Returns
        -------
        The recommended arm or recommended arms list.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Prediction with context policy requires context data.
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
            If contexts is not ``None`` for context-free bandits, the predicted expectations returned will be a
            list of the same length as contexts.

        Returns
        -------
        The dictionary of arms (key) to their expected rewards (value), or a list of such dictionaries.

        Raises
        ------
        TypeError:  Contexts is not given as ``None``, list, numpy array or pandas data frames.

        ValueError: Prediction with context policy requires context data.
        """

        # Check that fit is called before
        check_true(self._is_initial_fit, Exception("Call fit before prediction"))

        # Validate arguments
        self._validate_predict_args(contexts)

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts)

        # Return a dictionary from arms (key) to expectations (value)
        return self._imp.predict_expectations(contexts)

    def warm_start(self, arm_to_features: Dict[Arm, List[Num]], distance_quantile: float) -> None:
        """Warm-start untrained (cold) arms of the multi-armed bandit.

        Validates arguments and raises exceptions in case there are violations.

        The warm-start procedure depends on the learning and neighborhood policy. Note that for certain neighborhood
        policies (e.g., LSHNearest, KNearest, Radius) warm start can only be performed after the nearest neighbors
        have been determined in the "predict" step. Accordingly, warm start has to be executed for each context being
        predicted which is computationally expensive.

        Parameters
        ----------
        arm_to_features : Dict[Arm, List[Num]]
            Numeric representation for each arm.
        distance_quantile : float
            Value between 0 and 1 used to determine if an item can be warm started or not using closest item.
            All cold items will be warm started if 1 and none will be warm started if 0.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Arm features are not given as a dictionary.
        TypeError:  Distance quantile is not given as a float.

        ValueError:  Distance quantile is not between 0 and 1.
        ValueError:  The arms in arm_to_features do not match arms.
        """
        check_true(isinstance(arm_to_features, dict), TypeError("Arm features are not given as a dictionary."))
        check_true(isinstance(distance_quantile, float), TypeError("Distance quantile is not given as a float."))
        check_true(0 <= distance_quantile <= 1, ValueError("Distance quantile is not between 0 and 1."))
        check_true(set(self.arms) == set(arm_to_features.keys()),
                   ValueError("The arms in arm features do not match arms."))
        self._imp.warm_start(arm_to_features, distance_quantile)

    @staticmethod
    def _validate_mab_args(arms, learning_policy, neighborhood_policy, seed, n_jobs, backend):
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
                               LearningPolicy.LinGreedy, LearningPolicy.LinTS, LearningPolicy.LinUCB)),
                   TypeError("Learning Policy type mismatch."))


        # Contextual Policy
        if neighborhood_policy:
            check_true(isinstance(neighborhood_policy,
                                  (NeighborhoodPolicy.Clusters, NeighborhoodPolicy.KNearest,
                                   NeighborhoodPolicy.LSHNearest, NeighborhoodPolicy.Radius,
                                   NeighborhoodPolicy.TreeBandit)),
                       TypeError("Context Policy type mismatch."))

            # Tree-Bandit learning policy compatibility
            if isinstance(neighborhood_policy, NeighborhoodPolicy.TreeBandit):
                check_true(is_compatible(learning_policy),
                           ValueError(
                               "Tree-Bandit is not compatible with the learning policy: " + str(learning_policy)))

        # Seed
        check_true(isinstance(seed, int), TypeError("The seed must be an integer."))

        # Parallel jobs
        check_true(isinstance(n_jobs, int), TypeError("Number of parallel jobs must be an integer."))
        check_true(n_jobs != 0, ValueError('Number of parallel jobs cannot be zero.'))
        if backend is not None:
            check_true(isinstance(backend, str), TypeError("Parallel backend must be a string."))

    def _validate_fit_args(self, decisions, rewards, contexts):
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

    def _validate_predict_args(self, contexts):
        """"
        Validates argument types for predict and predict_expectation functions.
        """

        # Context policy and context data should match
        if self.is_contextual:  # don't use "if contexts" since it's n-dim array
            check_true(contexts is not None, ValueError("Prediction with context policy requires context data."))
            MAB._validate_context_type(contexts)
        else:
            if contexts is not None:
                MAB._validate_context_type(contexts)

    @staticmethod
    def _validate_context_type(contexts):
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

    @staticmethod
    def _validate_arm(arm):
        """
        Validates new arm.
        """
        check_false(arm is None, ValueError("The arm cannot be None."))
        check_false(np.nan in [arm], ValueError("The arm cannot be NaN."))
        check_false(np.inf in [arm], ValueError("The arm cannot be Infinity."))

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
                if isinstance(self.learning_policy, (LearningPolicy.LinGreedy,
                                                     LearningPolicy.LinTS,
                                                     LearningPolicy.LinUCB)):
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
