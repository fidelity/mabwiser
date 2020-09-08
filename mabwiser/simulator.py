# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides a simulation utility for comparing algorithms and hyper-parameter tuning.
"""

import abc
import logging
from copy import deepcopy
from collections import defaultdict
from itertools import chain
from typing import Union, List, Optional, NoReturn

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mabwiser.base_mab import BaseMAB
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.mab import MAB
from mabwiser.neighbors import _Neighbors, _Radius, _KNearest
from mabwiser.approximate import _LSHNearest
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, Num, check_true, Constants, _BaseRNG, create_rng
from mabwiser._version import __author__, __email__, __version__, __copyright__

__author__ = __author__
__email__ = __email__
__version__ = __version__
__copyright__ = __copyright__


def default_evaluator(arms: List[Arm], decisions: np.ndarray, rewards: np.ndarray, predictions: List[Arm],
                      arm_to_stats: dict, stat: str, start_index: int, nn: bool = False) -> dict:
    """Default evaluation function.

    Calculates predicted rewards for the test batch based on predicted arms.
    When the predicted arm is the same as the historic decision, the historic reward is used.
    When the predicted arm is different, the mean, min or max reward from the training data is used.
    If using Radius or KNearest neighborhood policy, the statistics from the neighborhood are used
    instead of the entire training set.

    The simulator supports custom evaluation functions,
    but they must have this signature to work with the simulation pipeline.

    Parameters
    ----------
    arms: list
        The list of arms.
    decisions: np.ndarray
        The historic decisions for the batch being evaluated.
    rewards: np.ndarray
        The historic rewards for the batch being evaluated.
    predictions: list
        The predictions for the batch being evaluated.
    arm_to_stats: dict
        The dictionary of descriptive statistics for each arm to use in evaluation.
    stat: str
        Which metric from arm_to_stats to use. Takes the values 'min', 'max', 'mean'.
    start_index: int
        The index of the first row in the batch.
        For offline simulations it is 0.
        For _online simulations it is batch size * batch number.
        Used to select the correct index from arm_to_stats if there are separate entries for each row in the test set.
    nn: bool
        Whether the results are from one of the simulator custom nearest neighbors implementations.

    Returns
    -------
    An arm_to_stats dictionary for the predictions in the batch.
    Dictionary has the format {arm {'count', 'sum', 'min', 'max', 'mean', 'std'}}
    """
    # If decision and prediction matches each other, use the observed reward
    # If decision and prediction are different, use the given stat (e.g., mean) for the arm as the reward

    arm_to_rewards = dict((arm, []) for arm in arms)
    if nn:
        arm_to_stats, neighborhood_stats = arm_to_stats
    for index, predicted_arm in enumerate(predictions):

        if predicted_arm == decisions[index]:
            arm_to_rewards[predicted_arm].append(rewards[index])
        elif nn:
            nn_index = index + start_index
            row_neighborhood_stats = neighborhood_stats[nn_index]
            if row_neighborhood_stats and row_neighborhood_stats[predicted_arm]:
                arm_to_rewards[predicted_arm].append(row_neighborhood_stats[predicted_arm][stat])
            else:
                arm_to_rewards[predicted_arm].append(arm_to_stats[predicted_arm][stat])

        else:
            arm_to_rewards[predicted_arm].append(arm_to_stats[predicted_arm][stat])

    # Calculate stats based on the rewards from predicted arms
    arm_to_stats_prediction = {}
    for arm in arms:
        arm_to_rewards[arm] = np.array(arm_to_rewards[arm])
        if len(arm_to_rewards[arm]) > 0:
            arm_to_stats_prediction[arm] = {'count': arm_to_rewards[arm].size, 'sum': arm_to_rewards[arm].sum(),
                                            'min': arm_to_rewards[arm].min(), 'max': arm_to_rewards[arm].max(),
                                            'mean': arm_to_rewards[arm].mean(), 'std': arm_to_rewards[arm].std()}
        else:
            arm_to_stats_prediction[arm] = {'count': 0, 'sum': math.nan,
                                            'min': math.nan, 'max': math.nan,
                                            'mean': math.nan, 'std': math.nan}

    return arm_to_stats_prediction


class _NeighborsSimulator(_Neighbors):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 metric: str, is_quick: bool, no_nhood_prob_of_arm: Optional[List] = None):
        super().__init__(rng, arms, n_jobs, backend, lp, metric, no_nhood_prob_of_arm)

        self.is_quick = is_quick
        self.neighborhood_arm_to_stat = []
        self.raw_rewards = None
        self.row_arm_to_expectation = []
        self.distances = None
        self.is_contextual = True
        self.neighborhood_sizes = []

    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None):
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
            self.raw_rewards = rewards.copy()

        super().fit(decisions, rewards, contexts)

    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None):
        if isinstance(self.lp, _ThompsonSampling) and self.lp.binarizer:
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
                                     contexts[starts[i]:starts[i + 1]])
                             for i in range(n_jobs))

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
            distances[index] = cdist(self.contexts, row_2d, metric=self.metric).reshape(-1)
        return distances

    def _predict_operation(self, contexts, is_predict):
        # Return predict within the neighborhood
        out = self._parallel_predict(contexts, is_predict=is_predict)

        if isinstance(out[0], list):
            df = pd.DataFrame(out, columns=['prediction', 'expectations', 'size', 'stats'])

            if is_predict:
                self.row_arm_to_expectation = self.row_arm_to_expectation + df['expectations'].tolist()
            else:
                self.row_arm_to_expectation = self.row_arm_to_expectation + df['prediction'].tolist()
            if not self.is_quick:
                self.neighborhood_sizes = self.neighborhood_sizes + df['size'].tolist()
                self.neighborhood_arm_to_stat = self.neighborhood_arm_to_stat + df['stats'].tolist()

            return df['prediction'].tolist()

        # Single row prediction
        else:
            prediction, expectation, size, stats = out
            if is_predict:
                self.row_arm_to_expectation = self.row_arm_to_expectation + [expectation]
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
                    arm_to_stat[arm] = Simulator.get_stats(arm_rewards)
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

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 radius: Num, metric: str, is_quick: bool, no_nhood_prob_of_arm: Optional[List] = None):
        super().__init__(rng, arms, n_jobs, backend, lp, metric, is_quick, no_nhood_prob_of_arm)
        self.radius = radius

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

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

                prediction, exp, stats = self._get_nhood_predictions(lp, row_2d, indices, is_predict)
                predictions[index] = [prediction, exp, len(indices[0]), stats]

            else:  # When there are no neighbors

                # Random arm (or nan expectations)
                prediction = self._get_no_nhood_predictions(lp, is_predict)
                predictions[index] = [prediction, {}, 0, {}]

        # Return the list of predictions
        return predictions


class _KNearestSimulator(_NeighborsSimulator):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 k: int, metric: str, is_quick: bool):
        super().__init__(rng, arms, n_jobs, backend, lp, metric, is_quick)
        self.k = k

    def _predict_contexts(self, contexts: np.ndarray, is_predict: bool,
                          seeds: Optional[np.ndarray] = None, start_index: Optional[int] = None) -> List:

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
            indices = np.argpartition(distances_to_row, self.k - 1)[:self.k]

            prediction, exp, stats = self._get_nhood_predictions(lp, row_2d, indices, is_predict)
            predictions[index] = [prediction, exp, self.k, stats]

        # Return the list of predictions
        return predictions


class _ApproximateSimulator(_NeighborsSimulator, metaclass=abc.ABCMeta):
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

                prediction, exp, stats = self._get_nhood_predictions(lp, row_2d, indices, is_predict)
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
    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, backend: Optional[str],
                 lp: Union[_EpsilonGreedy, _Linear, _Popularity, _Random, _Softmax, _ThompsonSampling, _UCB1],
                 n_dimensions: int, n_tables: int, is_quick: bool, no_nhood_prob_of_arm: Optional[List] = None):
        super().__init__(rng, arms, n_jobs, backend, lp, 'simhash', is_quick, no_nhood_prob_of_arm)

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
                delayed(_LSHNearest.get_context_hash)(
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

    def _initialize(self, n_rows):
        self.table_to_plane = {i: self.rng.standard_normal(size=(n_rows, self.n_dimensions))
                               for i in self.table_to_plane.keys()}

    def _get_neighbors(self, row_2d):
        indices = list()

        # Get list of neighbors from each hash table based on the hash values of the new context
        for k in self.table_to_plane.keys():
            hash_value = _LSHNearest.get_context_hash(row_2d, self.table_to_plane[k])
            indices += self.table_to_hash_to_index[k][hash_value[0]]
        return indices


class Simulator:
    """ Multi-Armed Bandit Simulator.

    This utility runs a simulation using historic data and a collection of multi-armed bandits from the MABWiser
    library or that extends the BaseMAB class in MABWiser.

    It can be used to run a simple simulation with a single bandit or to compare multiple bandits for policy selection,
    hyper-parameter tuning, etc.

    Nearest Neighbor bandits that use the default Radius and KNearest implementations from MABWiser are converted to
    custom versions that share distance calculations to speed up the simulation. These custom versions also track
    statistics about the neighborhoods that can be used in evaluation.

    The results can be accessed as the arms_to_stats, model_to_predictions, model_to_confusion_matrices, and
    models_to_evaluations properties.

    When using partial fitting, an additional confusion matrix is calculated for all predictions after all of the
    batches are processed.

    A log of the simulation tracks the experiment progress.

    Attributes
    ----------
    bandits: list[(str, bandit)]
        A list of tuples of the name of each bandit and the bandit object.
    decisions: array
        The complete decision history to be used in train and test.
    rewards: array
        The complete array history to be used in train and test.
    contexts: array
        The complete context history to be used in train and test.
    scaler: scaler
        A scaler object from sklearn.preprocessing.
    test_size: float
        The size of the test set
    is_ordered: bool
        Whether to use a chronological division for the train-test split.
        If false, uses sklearn's train_test_split.
    batch_size: int
        The size of each batch for online learning.
    evaluator: callable
        The function for evaluating the bandits. Values are stored in bandit_to_arm_to_stats_avg.
        Must have the function signature function(arms_to_stats_train: dictionary, predictions: list,
        decisions: np.ndarray, rewards: np.ndarray, metric: str).
    is_quick: bool
        Flag to skip neighborhood statistics.
    logger: Logger
        The logger object.
    arms: list
        The list of arms used by the bandits.
    arm_to_stats_total: dict
        Descriptive statistics for the complete data set.
    arm_to_stats_train: dict
        Descriptive statistics for the training data.
    arm_to_stats_test: dict
        Descriptive statistics for the test data.
    bandit_to_arm_to_stats_avg: dict
        Descriptive statistics for the predictions made by each bandit based on means from training data.
    bandit_to_arm_to_stats_min: dict
        Descriptive statistics for the predictions made by each bandit based on minimums from training data.
    bandit_to_arm_to_stats_max: dict
        Descriptive statistics for the predictions made by each bandit based on maximums from training data.
    bandit_to_confusion_matrices: dict
        The confusion matrices for each bandit.
    bandit_to_predictions: dict
        The prediction for each item in the test set for each bandit.
    bandit_to_expectations: dict
        The arm_to_expectations for each item in the test set for each bandit.
        For context-free bandits, there is a single dictionary for each batch.
    bandit_to_neighborhood_size: dict
        The number of neighbors in each neighborhood for each row in the test set.
        Calculated when using a Radius neighborhood policy, or a custom class that inherits from it.
        Not calculated when is_quick is True.
    bandit_to_arm_to_stats_neighborhoods: dict
        The arm_to_stats for each neighborhood for each row in the test set.
        Calculated when using Radius or KNearest, or a custom class that inherits from one of them.
        Not calculated when is_quick is True.
    test_indices: list
        The indices of the rows in the test set.
        If input was not zero-indexed, these will reflect their position in the input rather than actual index.

    Example
    -------
        >>> from mabwiser.mab import MAB, LearningPolicy
        >>> arms = ['Arm1', 'Arm2']
        >>> decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        >>> rewards = [20, 17, 25, 9]
        >>> mab1 = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.25), seed=123456)
        >>> mab2 = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0.30), seed=123456)
        >>> bandits = [('EG 25%', mab1), ('EG 30%', mab2)]
        >>> offline_sim = Simulator(bandits, decisions, rewards, test_size=0.5, batch_size=0)
        >>> offline_sim.run()
        >>> offline_sim.bandit_to_arm_to_stats_avg['EG 30%']['Arm1']
        {'count': 1, 'sum': 9, 'min': 9, 'max': 9, 'mean': 9.0, 'std': 0.0}

    """

    def __init__(self, bandits: List[tuple],                                    # List of tuples of names and bandits
                 decisions: Union[List[Arm], np.ndarray, pd.Series],            # Decisions that are made
                 rewards: Union[List[Num], np.ndarray, pd.Series],              # Rewards that are received
                 contexts: Union[None, List[List[Num]],
                                 np.ndarray, pd.Series, pd.DataFrame] = None,   # Contexts, optional
                 scaler: callable = None,                                       # Scaler for contexts
                 test_size: float = 0.3,                                        # Fraction to use for test batch
                 is_ordered: bool = False,                                      # Whether to use chronological order
                 batch_size: int = 0,                                           # Batch size for online learning
                 evaluator: callable = default_evaluator,                       # Evaluator function
                 seed: int = Constants.default_seed,                            # Random seed
                 is_quick: bool = False,                                        # Quick run flag
                 log_file: str = None,                                          # Log file name
                 log_format: str = '%(asctime)s %(levelname)s %(message)s'):    # Log file format
        """Simulator

        Creates a simulator object with a collection of bandits, the history of decisions, rewards, and contexts, and
        the parameters for the simulation.

        Parameters
        ----------
        bandits: list[tuple(str, MAB)]
            The set of bandits to run the simulation with. Must be a list of tuples of an identifier for the bandit and
            the bandit object, of type mabwiser.mab.MAB or that inherits from mabwiser.base_mab.BaseMAB
        decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
        rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
        contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is None.
        scaler: scaler
            One of the scalers from sklearn.preprocessing. Optional.
        test_size: float
            The fraction of data to use in the test set. Must be in the range (0, 1).
        is_ordered: bool
            Whether to divide the data randomly or to use the order given.
            When set to True, the test data will be the final n rows of the data set where n is determined by the split.
            When set to False, sklearn's train_test_split will be used.
        batch_size: int
            The batch size to test before partial fitting during _online learning.
            Cannot exceed the size of the test set.
            When batch size is 0, the simulation will be offline.
        evaluator: callable
            Function for scoring the predictions.
            Must have the function signature function(arm_to_stats_train: dictionary, predictions: list,
            decisions: np.ndarray, rewards: np.ndarray, stat: str, start_index: int, nn: bool).
        seed: num
            The seed for simulation
        is_quick: bool
            Flag to omit neighborhood statistics.
            Default value is False.
        log_file: str
            The logfile to store debug output. Optional.
        log_format: str
            The logger format used

        Raises
        ------
        TypeError   The bandit objects must be given in a list.
        TypeError   Each bandit object must be identified by a string label.
        TypeError   Each bandit must be of type MAB or inherit from BaseMAB.
        TypeError   The decisions must be given in a list, numpy array, or pandas Series.
        TypeError   The rewards must be given in a list, numpy array, or pandas series.
        TypeError   The contexts must be given in a 2D list, numpy array, pandas dataframe or pandas series.
        TypeError   The test_size size must be a float.
        TypeError   The batch size must be an integer.
        TypeError   The is_ordered flag must be a boolean.
        TypeError   The evaluation function must be callable.
        ValueError  The length of decisions and rewards must match.
        ValueError  The test_size size must be greater than 0 and less than 1.
        ValueError  The batch size cannot exceed the size of the test set.
        """

        self._validate_args(bandits=bandits, decisions=decisions, rewards=rewards, contexts=contexts,
                            test_size=test_size, ordered=is_ordered, batch_size=batch_size,
                            evaluation=evaluator, is_quick=is_quick)

        # Convert decisions, rewards and contexts to numpy arrays
        decisions = MAB._convert_array(decisions)
        rewards = MAB._convert_array(rewards)
        contexts = MAB._convert_matrix(contexts)

        # Save the simulation parameters
        self.bandits = bandits
        self.decisions = decisions
        self.rewards = rewards
        self.contexts = contexts
        self.scaler = scaler
        self.test_size = test_size
        self.is_ordered = is_ordered
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.seed = seed
        self.is_quick = is_quick
        self.log_file = log_file
        self.log_format = log_format

        self._online = batch_size > 0
        self._chunk_size = 100

        # logger object
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(self.log_format)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # create error file handler and set level to debug
        if self.log_file is not None:
            handler = logging.FileHandler(self.log_file, "w", encoding=None, delay="true")
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(self.log_format)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # set arms
        iter_name, iter_mab = self.bandits[0]
        self.arms = iter_mab.arms

        # Get the number of effective jobs for each bandit
        n_jobs_list = [BaseMAB._effective_jobs(math.ceil((len(decisions) * test_size)), mab.n_jobs)
                       for mab_name, mab in self.bandits]
        # set max n_jobs
        self.max_n_jobs = max(n_jobs_list)

        # Initialize statistic objects
        self.arm_to_stats_total = {}
        self.arm_to_stats_train = {}
        self.arm_to_stats_test = {}
        self.bandit_to_arm_to_stats_min = {}
        self.bandit_to_arm_to_stats_avg = {}
        self.bandit_to_arm_to_stats_max = {}
        self.bandit_to_confusion_matrices = {}

        # Test row metrics
        self.bandit_to_predictions = {}
        self.bandit_to_expectations = {}
        self.bandit_to_neighborhood_size = {}
        self.bandit_to_arm_to_stats_neighborhoods = {}
        self.test_indices = []

        # Log parameters
        self.logger.info('Simulation Parameters')
        self.logger.info("\t bandits: " + str(self.bandits))
        self.logger.info("\t scaler: " + str(self.scaler))
        self.logger.info("\t test_size: " + str(self.test_size))
        self.logger.info("\t is_ordered: " + str(self.is_ordered))
        self.logger.info("\t batch_size: " + str(self.batch_size))
        self.logger.info("\t evaluator: " + str(self.evaluator))
        self.logger.info("\t seed: " + str(self.seed))
        self.logger.info("\t is_quick: " + str(self.is_quick))
        self.logger.info("\t log_file: " + str(self.log_file))
        self.logger.info("\t format: " + self.log_format)

    # Public Methods
    def get_arm_stats(self, decisions: np.ndarray, rewards: np.ndarray) -> dict:
        """
        Calculates descriptive statistics for each arm in the provided data set.

        Parameters
        ----------
        decisions: np.ndarray
            The decisions to filter the rewards.
        rewards: np.ndarray
            The rewards to get statistics about.

        Returns
        -------
        Arm_to_stats dictionary.
        Dictionary has the format {arm {'count', 'sum', 'min', 'max', 'mean', 'std'}}
        """
        stats = dict((arm, {}) for arm in self.arms)
        for arm in self.arms:
            indices = np.where(decisions == arm)
            if indices[0].shape[0] > 0:
                arm_rewards = rewards[indices]
                stats[arm] = self.get_stats(arm_rewards)
            else:
                stats[arm] = {'count': 0, 'sum': 0, 'min': 0,
                              'max': 0, 'mean': 0, 'std': 0}
                self.logger.info('No historic data for ' + str(arm))
        return stats

    def plot(self, metric: str = 'avg', is_per_arm: bool = False) -> NoReturn:
        """
        Generates a plot of the cumulative sum of the rewards for each bandit.
        Simulation must be run before calling this method.

        Arguments
        ---------
        metric: str
            The bandit_to_arm_to_stats to use to generate the plot. Must be 'avg', 'min', or 'max
        is_per_arm: bool
            Whether to plot each arm separately or use an aggregate statistic.

        Raises
        ------
        AssertionError  Descriptive statics for predictions are missing.
        TypeError       Metric must be a string.
        TypeError       The per_arm flag must be a boolean.
        ValueError      The metric must be one of avg, min or max.

        Returns
        -------
        None
        """
        # Validate args
        check_true(isinstance(metric, str), TypeError('Metric must be a string.'))
        check_true(metric in ['avg', 'min', 'max'], ValueError('Metric must be one of avg, min or max.'))
        check_true(isinstance(is_per_arm, bool), TypeError('is_per_arm must be True or False.'))

        # Validate that simulation has been run
        complete = 'Complete simulation must be run before calling this method.'
        check_true(bool(self.bandit_to_arm_to_stats_min),
                   AssertionError('Descriptive statistics for predictions missing. ' + complete))

        if metric == 'avg':
            stats = self.bandit_to_arm_to_stats_avg
        elif metric == 'min':
            stats = self.bandit_to_arm_to_stats_min
        else:
            stats = self.bandit_to_arm_to_stats_max

        if self.batch_size > 0:
            cu_sums = {}
            labels = {}
            mabs = []

            if is_per_arm:
                for mab_name, mab in self.bandits:
                    self.logger.info('Plotting ' + str(mab_name))
                    for arm in self.arms:
                        mab_arm_name = str(mab_name) + '_' + str(arm)
                        mabs.append(mab_arm_name)
                        labels[mab_arm_name] = []
                        sums = []
                        cu_sums[mab_arm_name] = []
                        for key in stats[mab_name].keys():
                            if key != 'total':
                                labels[mab_arm_name].append(key)
                                if np.isnan(stats[mab_name][key][arm]['sum']):
                                    sums.append(0)
                                else:
                                    sums.append(stats[mab_name][key][arm]['sum'])
                        cs = 0
                        for item in sums:
                            cs += item
                            cu_sums[mab_arm_name].append(cs)
            else:
                for mab_name, mab in self.bandits:
                    self.logger.info('Plotting ' + str(mab_name))

                    mabs.append(mab_name)
                    labels[mab_name] = []
                    sums = []
                    cu_sums[mab_name] = []

                    for key in stats[mab_name].keys():
                        if key != 'total':

                            labels[mab_name].append(key)

                            net = 0
                            for arm in self.arms:
                                if np.isnan(stats[mab_name][key][arm]['sum']):
                                    continue

                                net += stats[mab_name][key][arm]['sum']
                            sums.append(net)
                    cs = 0

                    for item in sums:
                        cs += item
                        cu_sums[mab_name].append(cs)

            x = [i * self.batch_size for i in labels[mabs[0]]]
            for mab in mabs:
                sns.lineplot(x=x, y=cu_sums[mab], label=mab)
            plt.xlabel('Test Rows Predicted')
            plt.ylabel('Cumulative Reward')
            plt.show()

        else:
            x_labels = []
            y_values = []

            if is_per_arm:
                for mab_name, mab in self.bandits:
                    for arm in self.arms:
                        x_labels.append(str(mab_name) + '_' + str(arm))
                        if not np.isnan(stats[mab_name][arm]['sum']):
                            y_values.append(stats[mab_name][arm]['sum'])
                        else:
                            y_values.append(0)

            else:
                for mab_name, mab in self.bandits:
                    x_labels.append(mab_name)
                    cumulative = 0
                    for arm in self.arms:
                        if not np.isnan(stats[mab_name][arm]['sum']):
                            cumulative += stats[mab_name][arm]['sum']
                    y_values.append(cumulative)

            plt.bar(x_labels, y_values)
            plt.xlabel('Bandit')
            plt.ylabel('Cumulative Reward')
            plt.xticks(rotation=45)
            plt.show()

        plt.close('all')

    def run(self) -> NoReturn:
        """ Run simulator

        Runs a simulation concurrently for all bandits in the bandits list.

        Returns
        -------
        None
        """

        #####################################
        # Total Stats
        #####################################
        self.logger.info("\n")
        self._set_stats("total", self.decisions, self.rewards)

        #####################################
        # Train-Test Split
        #####################################
        self.logger.info("\n")
        self.logger.info("Train/Test Split")
        train_decisions, train_rewards, train_contexts, test_decisions, test_rewards, test_contexts = \
            self._run_train_test_split()

        self.logger.info('Train size: ' + str(len(train_decisions)))
        self.logger.info('Test size: ' + str(len(test_decisions)))

        #####################################
        # Scale the Data
        #####################################
        if self.scaler is not None:
            self.logger.info("\n")
            train_contexts, test_contexts = self._run_scaler(train_contexts, test_contexts)

        #####################################
        # Train/Test Stats
        #####################################
        self.logger.info("\n")
        self._set_stats("train", train_decisions, train_rewards)

        self.logger.info("\n")
        self._set_stats("test", test_decisions, test_rewards)

        #####################################
        # Fit the Training Data
        #####################################
        self.logger.info("\n")
        self._train_bandits(train_decisions, train_rewards, train_contexts)

        #####################################
        # Test the bandit simulation
        #####################################
        self.logger.info("\n")
        self.logger.info("Testing Bandits")
        if self._online:
            self._online_test_bandits(test_decisions, test_rewards, test_contexts)

        # If not running an _online simulation, evaluate the entire test set
        else:
            self._offline_test_bandits(test_decisions, test_rewards, test_contexts)

        self.logger.info('Simulation complete')

    # Private Methods
    def _get_partial_evaluation(self, name, i, decisions, predictions, rewards, start_index, nn=False):
        cfm = confusion_matrix(decisions, predictions)
        self.bandit_to_confusion_matrices[name].append(cfm)
        self.logger.info(str(name) + ' batch ' + str(i) + ' confusion matrix: ' + str(cfm))
        if nn and not self.is_quick:
            self.bandit_to_arm_to_stats_min[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      (self.arm_to_stats_train,
                                                                       self.bandit_to_arm_to_stats_neighborhoods[
                                                                           name]),
                                                                      "min", start_index, nn)

            self.bandit_to_arm_to_stats_avg[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      (self.arm_to_stats_train,
                                                                       self.bandit_to_arm_to_stats_neighborhoods[
                                                                           name]),
                                                                      "mean", start_index, nn)

            self.bandit_to_arm_to_stats_max[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      (self.arm_to_stats_train,
                                                                       self.bandit_to_arm_to_stats_neighborhoods[
                                                                           name]),
                                                                      "max", start_index, nn)
        else:
            self.bandit_to_arm_to_stats_min[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      self.arm_to_stats_train, "min",
                                                                      start_index, False)

            self.bandit_to_arm_to_stats_avg[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      self.arm_to_stats_train, "mean",
                                                                      start_index, False)

            self.bandit_to_arm_to_stats_max[name][i] = self.evaluator(self.arms,
                                                                      decisions, rewards,
                                                                      predictions,
                                                                      self.arm_to_stats_train, "max",
                                                                      start_index, False)
        self.logger.info(name + ' ' + str(self.bandit_to_arm_to_stats_min[name][i]))
        self.logger.info(name + ' ' + str(self.bandit_to_arm_to_stats_avg[name][i]))
        self.logger.info(name + ' ' + str(self.bandit_to_arm_to_stats_max[name][i]))

    def _offline_test_bandits(self, test_decisions, test_rewards, test_contexts):
        """
        Performs offline prediction.

        Arguments
        ---------
        test_decisions: np.ndarray
            The test set decisions.
        test_rewards: np.ndarray
            The test set rewards.
        test_contexts: np.ndarray
            The test set contexts.
        """

        chunk_start_index = [idx for idx in range(int(math.ceil(len(test_decisions) / self._chunk_size)))]
        for idx in chunk_start_index:

            # Set distances to None for new chunk
            distances = None

            # Progress update
            self.logger.info("Chunk " + str(idx + 1) + " out of " + str(len(chunk_start_index)))

            start = idx * self._chunk_size
            stop = min((idx+1)*self._chunk_size, len(test_decisions))
            chunk_decision = test_decisions[start:stop]
            chunk_contexts = test_contexts[start:stop] if test_contexts is not None else None

            for name, mab in self.bandits:

                if mab.is_contextual:
                    if isinstance(mab, (_RadiusSimulator, _KNearestSimulator)):
                        if distances is None:
                            distances = mab.calculate_distances(chunk_contexts)
                        else:
                            mab.set_distances(distances)
                        predictions = mab.predict(chunk_contexts)
                        expectations = mab.row_arm_to_expectation[start:stop].copy()

                    else:
                        predictions = mab.predict(chunk_contexts)
                        if isinstance(mab, _LSHSimulator):
                            expectations = mab.row_arm_to_expectation[start:stop].copy()
                        elif isinstance(mab._imp, _Neighbors):
                            expectations = mab._imp.arm_to_expectation.copy()
                        else:
                            expectations = mab.predict_expectations(chunk_contexts)

                    if not isinstance(expectations, list):
                        expectations = [expectations]
                    self.bandit_to_expectations[name] = self.bandit_to_expectations[name] + expectations

                else:
                    predictions = [mab.predict() for _ in range(len(chunk_decision))]

                if not isinstance(predictions, list):
                    predictions = [predictions]

                self.bandit_to_predictions[name] = self.bandit_to_predictions[name] + predictions

                if isinstance(mab, _NeighborsSimulator) and not self.is_quick:
                    self.bandit_to_arm_to_stats_neighborhoods[name] = mab.neighborhood_arm_to_stat.copy()

        for name, mab in self.bandits:
            nn = isinstance(mab, _NeighborsSimulator)

            if not mab.is_contextual:
                self.bandit_to_expectations[name] = mab._imp.arm_to_expectation.copy()
            if isinstance(mab, _NeighborsSimulator) and not self.is_quick:
                self.bandit_to_neighborhood_size[name] = mab.neighborhood_sizes.copy()

            # Evaluate the predictions
            self.bandit_to_confusion_matrices[name].append(confusion_matrix(test_decisions,
                                                                            self.bandit_to_predictions[name]))

            self.logger.info(name + " confusion matrix: " + str(self.bandit_to_confusion_matrices[name]))

            if nn and not self.is_quick:
                self.bandit_to_arm_to_stats_min[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       (self.arm_to_stats_train,
                                                                        self.bandit_to_arm_to_stats_neighborhoods[
                                                                            name]),
                                                                       stat="min", start_index=0, nn=nn)

                self.bandit_to_arm_to_stats_avg[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       (self.arm_to_stats_train,
                                                                        self.bandit_to_arm_to_stats_neighborhoods[
                                                                            name]),
                                                                       stat="mean", start_index=0, nn=nn)

                self.bandit_to_arm_to_stats_max[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       (self.arm_to_stats_train,
                                                                        self.bandit_to_arm_to_stats_neighborhoods[
                                                                            name]),
                                                                       stat="max", start_index=0, nn=nn)
            else:
                self.bandit_to_arm_to_stats_min[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       self.arm_to_stats_train, stat="min",
                                                                       start_index=0, nn=False)

                self.bandit_to_arm_to_stats_avg[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       self.arm_to_stats_train, stat="mean",
                                                                       start_index=0, nn=False)

                self.bandit_to_arm_to_stats_max[name] = self.evaluator(self.arms,
                                                                       test_decisions, test_rewards,
                                                                       self.bandit_to_predictions[name],
                                                                       self.arm_to_stats_train, stat="max",
                                                                       start_index=0, nn=False)

            self.logger.info(name + " minimum analysis " + str(self.bandit_to_arm_to_stats_min[name]))
            self.logger.info(name + " average analysis " + str(self.bandit_to_arm_to_stats_avg[name]))
            self.logger.info(name + " maximum analysis " + str(self.bandit_to_arm_to_stats_max[name]))

    def _online_test_bandits(self, test_decisions, test_rewards, test_contexts):
        """
        Performs _online prediction and partial fitting for each model.

        Arguments
        ---------
        test_decisions: np.ndarray
            The test set decisions.
        test_rewards: np.ndarray
            The test set rewards.
        test_contexts: np.ndarray
            The test set contexts.
        """

        # Divide the test data into batches and chunk the batches based on size
        self._online_test_bandits_chunks(test_decisions, test_rewards, test_contexts)

        # Final scores for all predictions
        for name, mab in self.bandits:
            nn = isinstance(mab, _NeighborsSimulator)

            self._get_partial_evaluation(name, 'total', test_decisions, self.bandit_to_predictions[name],
                                         test_rewards, 0, nn)

            if isinstance(mab, _NeighborsSimulator) and not self.is_quick:
                self.bandit_to_neighborhood_size[name] = mab.neighborhood_sizes.copy()
                self.bandit_to_arm_to_stats_neighborhoods[name] = mab.neighborhood_arm_to_stat.copy()

    def _online_test_bandits_chunks(self, test_decisions, test_rewards, test_contexts):
        """
        Performs _online prediction and partial fitting for each model.

        Arguments
        ---------
        test_decisions: np.ndarray
            The test set decisions.
        test_rewards: np.ndarray
            The test set rewards.
        test_contexts: np.ndarray
            The test set contexts.
        """

        # Divide the test data into batches
        start = 0
        for i in range(0, int(math.ceil(len(test_decisions) / self.batch_size))):
            self.logger.info('Starting batch ' + str(i))

            # Stop at the next batch_size interval or the end of the test data
            stop = min(start + self.batch_size, len(test_decisions) + 1)

            batch_contexts = test_contexts[start:stop] if test_contexts is not None else None
            batch_decisions = test_decisions[start:stop]
            batch_rewards = test_rewards[start:stop]
            batch_predictions = {}
            batch_expectations = {}

            chunk_start = 0

            # Divide the batch into chunks
            for j in range(0, int(math.ceil(self.batch_size / self._chunk_size))):
                distances = None
                chunk_stop = min(chunk_start + self._chunk_size, self.batch_size)
                chunk_contexts = batch_contexts[chunk_start:chunk_stop] if batch_contexts is not None else None
                chunk_decisions = batch_decisions[chunk_start:chunk_stop]

                for name, mab in self.bandits:

                    if name not in batch_predictions.keys():
                        batch_predictions[name] = []
                        batch_expectations[name] = []

                    # Predict for the batch
                    if mab.is_contextual:
                        if isinstance(mab, (_RadiusSimulator, _KNearestSimulator)):
                            if distances is None:
                                distances = mab.calculate_distances(chunk_contexts)
                                self.logger.info('Distances calculated')
                            else:
                                mab.set_distances(distances)
                                self.logger.info('Distances set')
                            predictions = mab.predict(chunk_contexts)
                            expectations = mab.row_arm_to_expectation[start+chunk_start:start+chunk_stop].copy()
                        else:
                            predictions = mab.predict(chunk_contexts)
                            if isinstance(mab, _LSHSimulator):
                                expectations = mab.row_arm_to_expectation[start:stop].copy()
                            else:
                                expectations = mab.predict_expectations(chunk_contexts)

                        if self.batch_size == 1:
                            predictions = [predictions]

                    else:
                        predictions = [mab.predict() for _ in range(len(chunk_decisions))]
                        expectations = mab._imp.arm_to_expectation.copy()

                    # If a single prediction was returned, put it into a list
                    if not isinstance(predictions, list):
                        predictions = [predictions]
                    if not isinstance(expectations, list):
                        expectations = [expectations]

                    batch_predictions[name] = batch_predictions[name] + predictions
                    batch_expectations[name] = batch_expectations[name] + expectations

            for name, mab in self.bandits:
                if not mab.is_contextual:
                    batch_expectations[name] = [mab._imp.arm_to_expectation.copy()]

                nn = isinstance(mab, _NeighborsSimulator)

                # Add predictions from this batch
                self.bandit_to_predictions[name] = self.bandit_to_predictions[name] + batch_predictions[name]
                self.bandit_to_expectations[name] = self.bandit_to_expectations[name] + batch_expectations[name]

                if isinstance(mab, (_RadiusSimulator, _LSHSimulator)) and not self.is_quick:
                    self.bandit_to_neighborhood_size[name] = mab.neighborhood_sizes.copy()
                if isinstance(mab, _NeighborsSimulator) and not self.is_quick:
                    self.bandit_to_arm_to_stats_neighborhoods[name] = mab.neighborhood_arm_to_stat.copy()

                # Evaluate the predictions
                self._get_partial_evaluation(name, i, batch_decisions, batch_predictions[name],
                                             batch_rewards, start, nn)

                # Update the model
                if mab.is_contextual:
                    mab.partial_fit(batch_decisions, batch_rewards, batch_contexts)
                else:
                    mab.partial_fit(batch_decisions, batch_rewards)
                self.logger.info(name + ' updated')

            # Update start value for next batch
            start += self.batch_size

    def _run_scaler(self, train_contexts, test_contexts):
        """
        Scales the train and test contexts with the scaler provided to the simulator constructor.

        Arguments
        ---------
        train_contexts: np.ndarray
            The training set contexts.
        test_contexts: np.ndarray
            The test set contexts.

        Returns
        -------
            The scaled train_contexts and test_contexts
        """

        self.logger.info("Train/Test Scale")

        train_contexts = self.scaler.fit_transform(train_contexts)
        test_contexts = self.scaler.transform(test_contexts)
        return train_contexts, test_contexts

    def _run_train_test_split(self):
        """
        Performs a train-test split with the test set containing a percentage of the data determined by test_size.

        If is_ordered is true, performs a chronological split.
        Otherwise uses sklearn's train_test_split

        Returns
        -------
            The train and test decisions, rewards and contexts
        """

        if self.is_ordered:
            train_size = int(len(self.decisions) * (1 - self.test_size))
            train_decisions = self.decisions[:train_size]
            train_rewards = self.rewards[:train_size]
            train_contexts = self.contexts[:train_size] if self.contexts is not None else None
            # The test arrays are re-indexed to 0 automatically
            test_decisions = self.decisions[train_size:]
            test_rewards = self.rewards[train_size:]
            test_contexts = self.contexts[train_size:] if self.contexts is not None else None
            self.test_indices = [x for x in range(train_size, len(self.decisions))]

        else:
            indices = [x for x in range(len(self.decisions))]
            if self.contexts is None:

                train_contexts, test_contexts = None, None

                train_indices, test_indices, train_decisions, test_decisions, train_rewards, test_rewards = \
                    train_test_split(indices, self.decisions, self.rewards, test_size=self.test_size,
                                     random_state=self.seed)
            else:

                train_indices, test_indices, train_decisions, test_decisions, train_rewards, test_rewards, \
                    train_contexts, test_contexts = \
                    train_test_split(indices, self.decisions, self.rewards, self.contexts,
                                     test_size=self.test_size, random_state=self.seed)
            self.test_indices = test_indices

        # Use memory limits for the nearest neighbors shared distance list to determine chunk size.
        # The list without chunking contains len(test_decisions) elements
        # each of which is an np.ndarray with len(train_decisions) distances.
        # Approximate as 8 bytes per element in each numpy array to give the size of the list in GB.
        distance_list_size = len(test_decisions) * (8 * len(train_decisions)) / 1e9

        # If there is more than one test row and contexts have been provided:
        if distance_list_size > 1.0 and train_contexts is not None:

            # Set the chunk size to contain 1GB per job
            gb_chunk_size = int(len(test_decisions) / distance_list_size) * self.max_n_jobs

            # If the length of the test set is less than the chunk size, chunking is unnecessary
            self._chunk_size = min(gb_chunk_size, len(test_decisions))

        # If there is only one test row or all MABs are context-free chunking is unnecessary:
        else:
            self._chunk_size = len(test_decisions)

        return train_decisions, train_rewards, train_contexts, test_decisions, test_rewards, test_contexts

    def _set_stats(self, scope, decisions, rewards):
        """
        Calculates descriptive statistics for each arm for the specified data set
        and stores them to the corresponding arm_to_stats dictionary.

        Arguments
        ---------
        scope: str
            The label for which set is being evaluated.
            Accepted values: 'total', 'train', 'test'
        decisions: np.ndarray
            The decisions to filter the rewards.
        rewards: np.ndarray
            The rewards to get statistics about.

        Returns
        -------
        None
        """

        if scope == 'total':
            self.arm_to_stats_total = self.get_arm_stats(decisions, rewards)
            self.logger.info("Total Stats")
            self.logger.info(self.arm_to_stats_total)
        elif scope == 'train':
            self.arm_to_stats_train = self.get_arm_stats(decisions, rewards)
            self.logger.info("Train Stats")
            self.logger.info(self.arm_to_stats_train)
        elif scope == 'test':
            self.arm_to_stats_test = self.get_arm_stats(decisions, rewards)
            self.logger.info("Test Stats")
            self.logger.info(self.arm_to_stats_test)
        else:
            raise ValueError("Unsupported scope name")

    def _train_bandits(self, train_decisions, train_rewards, train_contexts=None):
        """
        Trains each of the bandit models.

        Arguments
        ---------
        train_decisions: np.ndarray
            The training set decisions.
        train_rewards: np.ndarray
            The training set rewards.
        train_contexts: np.ndarray
            The training set contexts.
        """

        self.logger.info("Training Bandits")

        new_bandits = []
        for name, mab in self.bandits:
            # Add the current bandit
            self.bandit_to_predictions[name] = []
            self.bandit_to_expectations[name] = []
            self.bandit_to_neighborhood_size[name] = []
            self.bandit_to_arm_to_stats_neighborhoods[name] = []
            self.bandit_to_confusion_matrices[name] = []
            self.bandit_to_arm_to_stats_min[name] = {}
            self.bandit_to_arm_to_stats_avg[name] = {}
            self.bandit_to_arm_to_stats_max[name] = {}

            if isinstance(mab, MAB):
                imp = mab._imp
            else:
                imp = mab
            if isinstance(imp, _Radius):
                mab = _RadiusSimulator(imp.rng, imp.arms, imp.n_jobs, imp.backend, imp.lp, imp.radius,
                                       imp.metric, is_quick=self.is_quick,
                                       no_nhood_prob_of_arm=imp.no_nhood_prob_of_arm)

            elif isinstance(imp, _KNearest):
                mab = _KNearestSimulator(imp.rng, imp.arms, imp.n_jobs, imp.backend, imp.lp, imp.k,
                                         imp.metric, is_quick=self.is_quick)
            elif isinstance(imp, _LSHNearest):
                mab = _LSHSimulator(imp.rng, imp.arms, imp.n_jobs, imp.backend, imp.lp,
                                    imp.n_dimensions, imp.n_tables, is_quick=self.is_quick,
                                    no_nhood_prob_of_arm=imp.no_nhood_prob_of_arm)

            new_bandits.append((name, mab))
            if mab.is_contextual:
                mab.fit(train_decisions, train_rewards, train_contexts)
            else:
                mab.fit(train_decisions, train_rewards)
            self.logger.info(name + ' trained')

        self.bandits = new_bandits

    # Static Methods
    @staticmethod
    def get_stats(rewards: np.ndarray) -> dict:
        """Calculates descriptive statistics for the given array of rewards.

        Parameters
        ----------
        rewards: nd.nparray
            Array of rewards for a single arm.

        Returns
        -------
        A dictionary of descriptive statistics.
        Dictionary has the format {'count', 'sum', 'min', 'max', 'mean', 'std'}
        """
        return {'count': rewards.size, 'sum': rewards.sum(), 'min': rewards.min(),
                'max': rewards.max(), 'mean': rewards.mean(), 'std': rewards.std()}

    @staticmethod
    def _validate_args(bandits, decisions, rewards, contexts, test_size, ordered, batch_size,
                       evaluation, is_quick):
        """
        Validates the simulation parameters.
        """
        check_true(isinstance(bandits, list), TypeError('Bandits must be provided in a list.'))
        for pair in bandits:
            name, mab = pair
            check_true(isinstance(name, str), TypeError('All bandits must be identified by strings.'))
            check_true(isinstance(mab, (MAB, BaseMAB)),
                       TypeError('All bandits must be MAB objects or inherit from BaseMab.'))

        # Type check for decisions
        check_true(isinstance(decisions, (list, np.ndarray, pd.Series)),
                   TypeError("The decisions should be given as list, numpy array, or pandas series."))

        # Type check for rewards
        check_true(isinstance(rewards, (list, np.ndarray, pd.Series)),
                   TypeError("The rewards should be given as list, numpy array, or pandas series."))

        # Type check for contexts --don't use "if contexts" since it's n-dim array
        if contexts is not None:
            if isinstance(contexts, np.ndarray):
                check_true(contexts.ndim == 2,
                           TypeError("The contexts should be given as 2D list, numpy array, or pandas series or "
                                     "data frames."))
            elif isinstance(contexts, list):
                check_true(np.array(contexts).ndim == 2,
                           TypeError("The contexts should be given as 2D list, numpy array, or pandas series or "
                                     "data frames."))
            else:
                check_true(isinstance(contexts, (pd.Series, pd.DataFrame)),
                           TypeError("The contexts should be given as 2D list, numpy array, or pandas series or "
                                     "data frames."))

        # Length check for decisions and rewards
        check_true(len(decisions) == len(rewards), ValueError("Decisions and rewards should be same length."))

        check_true(isinstance(test_size, float), TypeError("Test size must be a float."))
        check_true(0.0 < test_size < 1.0, ValueError("Test size must be greater than zero and less than one."))
        check_true(isinstance(ordered, bool), TypeError("Ordered must be a boolean."))
        check_true(isinstance(batch_size, int), TypeError("Batch size must be an integer."))
        if batch_size > 0:
            check_true(batch_size <= (math.ceil(len(decisions) * test_size)),
                       ValueError("Batch size cannot be larger than " "the test set."))

        check_true(callable(evaluation), TypeError("Evaluation method must be a function."))

        check_true(isinstance(is_quick, bool), TypeError('Quick run flag must be a boolean.'))
