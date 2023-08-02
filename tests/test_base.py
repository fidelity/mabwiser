# -*- coding: utf-8 -*-

import unittest
from typing import List, Union, Optional

import numpy as np
import pandas as pd

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy, LearningPolicyType, NeighborhoodPolicyType
from mabwiser.utils import Arm, Num


class BaseTest(unittest.TestCase):

    # A list of valid learning policies
    lps = [LearningPolicy.EpsilonGreedy(),
           LearningPolicy.EpsilonGreedy(epsilon=0),
           LearningPolicy.EpsilonGreedy(epsilon=0.0),
           LearningPolicy.EpsilonGreedy(epsilon=0.5),
           LearningPolicy.EpsilonGreedy(epsilon=1),
           LearningPolicy.EpsilonGreedy(epsilon=1.0),
           LearningPolicy.Popularity(),
           LearningPolicy.Random(),
           LearningPolicy.Softmax(),
           LearningPolicy.Softmax(tau=0.1),
           LearningPolicy.Softmax(tau=0.5),
           LearningPolicy.Softmax(tau=1),
           LearningPolicy.Softmax(tau=1.0),
           LearningPolicy.Softmax(tau=5.0),
           LearningPolicy.ThompsonSampling(),
           LearningPolicy.UCB1(),
           LearningPolicy.UCB1(alpha=0),
           LearningPolicy.UCB1(alpha=0.0),
           LearningPolicy.UCB1(alpha=0.5),
           LearningPolicy.UCB1(alpha=1),
           LearningPolicy.UCB1(alpha=1.0),
           LearningPolicy.UCB1(alpha=5)]

    para_lps = [LearningPolicy.LinGreedy(epsilon=0, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=0.5, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=1, l2_lambda=1),
                LearningPolicy.LinGreedy(epsilon=0, l2_lambda=0.5),
                LearningPolicy.LinGreedy(epsilon=0.5, l2_lambda=0.5),
                LearningPolicy.LinGreedy(epsilon=1, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=0.00001, l2_lambda=1),
                LearningPolicy.LinTS(alpha=0.5, l2_lambda=1),
                LearningPolicy.LinTS(alpha=1, l2_lambda=1),
                LearningPolicy.LinTS(alpha=0.00001, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=0.5, l2_lambda=0.5),
                LearningPolicy.LinTS(alpha=1, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=0, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=0.5, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=1, l2_lambda=1),
                LearningPolicy.LinUCB(alpha=0, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=0.5, l2_lambda=0.5),
                LearningPolicy.LinUCB(alpha=1, l2_lambda=0.5)]

    # A list of valid context policies
    nps = [NeighborhoodPolicy.LSHNearest(),
           NeighborhoodPolicy.LSHNearest(n_dimensions=1),
           NeighborhoodPolicy.LSHNearest(n_dimensions=1, n_tables=1),
           NeighborhoodPolicy.KNearest(),
           NeighborhoodPolicy.KNearest(k=1),
           NeighborhoodPolicy.KNearest(k=3),
           NeighborhoodPolicy.Radius(),
           NeighborhoodPolicy.Radius(2.5),
           NeighborhoodPolicy.Radius(5),
           NeighborhoodPolicy.TreeBandit()]

    cps = [NeighborhoodPolicy.Clusters(),
           NeighborhoodPolicy.Clusters(n_clusters=3),
           NeighborhoodPolicy.Clusters(is_minibatch=True),
           NeighborhoodPolicy.Clusters(n_clusters=3, is_minibatch=True)]

    @staticmethod
    def predict(arms: List[Arm],
                decisions: Union[List, np.ndarray, pd.Series],
                rewards: Union[List, np.ndarray, pd.Series],
                learning_policy: LearningPolicyType,
                neighborhood_policy: NeighborhoodPolicyType = None,
                context_history: Union[None, List[Num], List[List[Num]], np.ndarray, pd.DataFrame, pd.Series] = None,
                contexts: Union[None, List[Num], List[List[Num]], np.ndarray, pd.DataFrame, pd.Series] = None,
                seed: Optional[int] = 123456,
                num_run: Optional[int] = 1,
                is_predict: Optional[bool] = True,
                n_jobs: Optional[int] = 1,
                backend: Optional[str] = None
                ) -> (Union[Arm, List[Arm], List[float], List[List[float]]], MAB):
        """Sets up a MAB model and runs the given configuration.

        Return list of predictions or prediction and the mab instance, when is_predict is true
        Return list of expectations or expectation and the mab instance, when is predict is false

        Calls the predict or predict_expectation method num_run number of times.
        """

        # Model
        mab = MAB(arms, learning_policy, neighborhood_policy, seed, n_jobs, backend)

        # Train
        mab.fit(decisions, rewards, context_history)

        # Test
        if is_predict:

            # Return: prediction(s) and the MAB instance
            predictions = [mab.predict(contexts) for _ in range(num_run)]
            return predictions[0] if num_run == 1 else predictions, mab

        else:

            # Return: expectations(s) and the MAB instance
            expectations = [mab.predict_expectations(contexts) for _ in range(num_run)]
            return expectations[0] if num_run == 1 else expectations, mab

    @staticmethod
    def is_compatible(learning_policy, neighborhood_policy):

        # Special case for TreeBandit lp/np compatibility
        if isinstance(neighborhood_policy, NeighborhoodPolicy.TreeBandit):
            return neighborhood_policy._is_compatible(learning_policy)

        return True

    def assertListAlmostEqual(self, list1, list2):
        """
        Asserts that floating values in the given lists (almost) equals to each other
        """
        if not isinstance(list1, list):
            list1 = list(list1)

        if not isinstance(list2, list):
            list2 = list(list2)

        self.assertEqual(len(list1), len(list2))

        for index, val in enumerate(list1):
            self.assertAlmostEqual(val, list2[index])
