# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from mabwiser.greedy import _EpsilonGreedy
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class ClustersTest(BaseTest):

    def test_greedy0_n2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])
        self.assertTrue(isinstance(mab._imp.kmeans, KMeans))

    def test_greedy0_n2_mini(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2, True),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])
        self.assertTrue(isinstance(mab._imp.kmeans, MiniBatchKMeans))

    def test_greedy0_n2_single(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, 3)

    def test_greedy0_n2_exps(self):

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertDictEqual(exps[0], {1: 0.0, 2: 0.0, 3: 0.5, 4: 0})
        self.assertDictEqual(exps[1], {1: 1.0, 2: 0.0, 3: 0.6666666666666666, 4: 0})

    def test_greedy0_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 2])
        self.assertEqual(mab._imp.kmeans.n_clusters, 3)

    def test_greedy1_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=1.0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_thompson_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_ucb_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(alpha=1),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])

    def test_softmax_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Softmax(tau=1),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 3])

    def test_partial_fit_greedy0_n3(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)

        decisions2 = [1, 2, 3]
        rewards2 = [1, 1, 1]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 13)
        self.assertEqual(len(mab._imp.rewards), 13)
        self.assertEqual(len(mab._imp.contexts), 13)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)

    def test_partial_fit_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp_list[0].is_contextual_binarized)
        self.assertListEqual(arms, [2, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertListEqual(list(set(mab._imp.rewards)), [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 13)
        self.assertEqual(len(mab._imp.rewards), 13)
        self.assertEqual(len(mab._imp.contexts), 13)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertListEqual(list(set(mab._imp.rewards)), [0, 1])

    def test_fit_twice_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(3),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp_list[0].is_contextual_binarized)
        self.assertListEqual(arms, [2, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertListEqual(list(set(mab._imp.rewards)), [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 3)
        self.assertEqual(len(mab._imp.rewards), 3)
        self.assertEqual(len(mab._imp.contexts), 3)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertListEqual(list(set(mab._imp.rewards)), [0, 1])

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        mab.add_arm(5)
        self.assertTrue(5 in mab.arms)
        self.assertTrue(5 in mab._imp.arms)
        self.assertTrue(5 in mab._imp.lp_list[0].arms)
        self.assertTrue(5 in mab._imp.lp_list[0].arm_to_expectation.keys())

    def test_copy(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        clusters = deepcopy(mab._imp)
        self.assertIsNot(clusters, mab._imp)
        self.assertIsInstance(clusters.lp_list[0], _EpsilonGreedy)
        self.assertIsInstance(clusters.lp_list[1], _EpsilonGreedy)
        self.assertIsInstance(clusters.kmeans, KMeans)
        self.assertIsNot(clusters.kmeans, mab._imp.kmeans)
        self.assertIsNot(clusters.lp_list[0], mab._imp.lp_list[0])
        self.assertIsNot(clusters.lp_list[1], mab._imp.lp_list[1])
        self.assertEqual(clusters.lp_list[0].epsilon, mab._imp.lp_list[0].epsilon)
        self.assertEqual(clusters.lp_list[1].epsilon, mab._imp.lp_list[1].epsilon)

    def test_copy_arms(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab.arms is mab._imp.lp_list[0].arms)
        self.assertTrue(mab.arms is mab._imp.lp_list[1].arms)
        mab.add_arm(5)
        self.assertTrue(mab.arms is mab._imp.lp_list[0].arms)
        self.assertTrue(mab.arms is mab._imp.lp_list[1].arms)

    def test_remove_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertTrue(mab.arms is mab._imp.lp_list[0].arms)
        self.assertTrue(mab.arms is mab._imp.lp_list[1].arms)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab._imp.lp_list[0].arms)
        self.assertTrue(3 not in mab._imp.lp_list[1].arms)

    def test_warm_start(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        # Before warm start
        self.assertEqual(mab._imp.lp_list[0].arms, [1, 2, 3, 4])
        self.assertEqual(mab._imp.lp_list[0].trained_arms, [1, 2, 3])
        self.assertDictEqual(mab._imp.lp_list[0].arm_to_expectation, {1: 1.0, 2: 0.0, 3: 0.6666666666666666, 4: 0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5], 4: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.lp_list[0].arm_to_expectation, {1: 1.0, 2: 0.0, 3: 0.6666666666666666, 4: 0})
