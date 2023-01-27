# -*- coding: utf-8 -*-

import numpy as np
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class RadiusTest(BaseTest):

    def test_greedy0_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])

    def test_greedy0_r2_single_test(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, 3)

    def test_greedy0_r2_single_list(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, 3)

    def test_greedy0_r2_exps(self):

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertDictEqual(exps[0], {1: 0.0, 2: 0.0, 3: 0.5, 4: 0})
        self.assertDictEqual(exps[1], {1: 1.0, 2: 0.0, 3: 1.0, 4: 0})

    def test_greedy0_r5(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(5),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 2])

    def test_greedy1_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=1.0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_thompson_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_ucb_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(alpha=1),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 3])

    def test_softmax_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Softmax(tau=1),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 3])

    def test_no_neighbors(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(.1),
                                 context_history=[[10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10]],
                                 contexts=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.01, .01, .01, .01, .01], [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1, 1, 4, 1])

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(.1),
                                 context_history=[[10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10]],
                                 contexts=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.01, .01, .01, .01, .01], [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]],
                                 seed=7,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 4, 1, 2, 2])

    def test_no_neighbors_expectations(self):

        exp, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(.1),
                                 context_history=[[10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10]],
                                 contexts=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.01, .01, .01, .01, .01], [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        for index, row in enumerate(exp):
            for key in row.keys():
                self.assertIs(np.nan, row[key])

        exp, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(.1),
                                 context_history=[[10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10],
                                                  [10, 10, 10, 10, 10]],
                                 contexts=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.01, .01, .01, .01, .01], [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]],
                                 seed=7,
                                 num_run=1,
                                 is_predict=False)

        for index, row in enumerate(exp):
            for key in row.keys():
                self.assertIs(np.nan, row[key])

    def test_partial_fit_greedy0_r2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
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
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp.is_contextual_binarized)
        self.assertListEqual(arms, [2, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 13)
        self.assertEqual(len(mab._imp.rewards), 13)
        self.assertEqual(len(mab._imp.contexts), 13)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

    def test_fit_twice_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp.is_contextual_binarized)
        self.assertListEqual(arms, [2, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 3)
        self.assertEqual(len(mab._imp.rewards), 3)
        self.assertEqual(len(mab._imp.contexts), 3)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
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
        self.assertTrue(5 in mab._imp.lp.arms)
        self.assertTrue(5 in mab._imp.lp.arm_to_expectation.keys())

    def test_greedy0_no_nhood_predict_random(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force Radius to no nhood
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(0.00001),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=2,
                                 is_predict=True)

        # 3rd arm was never seen but picked up by random neighborhood in both tests
        self.assertListEqual(arms[0], [3, 1])
        self.assertListEqual(arms[1], [1, 3])

    def test_greedy0_no_nhood_predict_weighted(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force Radius to no nhoods
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(0.00001,
                                                                               no_nhood_prob_of_arm=[0, 0.8, 0.2]),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=45676,
                                 num_run=2,
                                 is_predict=True)

        # 2nd arm is weighted highly but 3rd is picked too
        self.assertListEqual(arms[0], [2, 2])
        self.assertListEqual(arms[1], [3, 3])

    def test_greedy0_no_nhood_expectation_nan(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force Radius to no nhoods
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(0.00001),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        # When there are no neighborhoods, expectations will be nan
        self.assertDictEqual(arms[0], {1: np.nan, 2: np.nan, 3: np.nan})
        self.assertDictEqual(arms[1], {1: np.nan, 2: np.nan, 3: np.nan})

    def test_remove_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertTrue(mab.arms is mab._imp.arms)
        self.assertTrue(mab.arms is mab._imp.lp.arms)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.lp.arms)
        self.assertTrue(3 not in mab._imp.lp.arm_to_expectation)

    def test_warm_start(self):
        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        # Before warm start
        self.assertDictEqual(exps[0], {1: 0.0, 2: 0.0, 3: 0.5, 4: 0})
        self.assertDictEqual(exps[1], {1: 1.0, 2: 0.0, 3: 1.0, 4: 0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5], 4: [0, 1]}, distance_quantile=0.5)
        exps = mab.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        # After warm start
        self.assertDictEqual(exps[0], {1: 0.0, 2: 0.0, 3: 0.5, 4: 0})
        self.assertDictEqual(exps[1], {1: 1.0, 2: 0.0, 3: 1.0, 4: 0})
