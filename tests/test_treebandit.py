# -*- coding: utf-8 -*-

import pandas as pd

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class TreeBanditTest(BaseTest):

    def test_doctest_example(self):
        list_of_arms = ['Arm1', 'Arm2']
        decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        rewards = [20, 17, 25, 9]
        contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
        mab = MAB(list_of_arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.TreeBandit())
        mab.fit(decisions, rewards, contexts)
        result = mab.predict([[3, 2, 0, 1]])
        self.assertEqual(result, 'Arm2')

    def test_tree_parameters_default(self):

        arm, mab = self.predict(arms=['Arm1', 'Arm2'],
                                decisions=['Arm1', 'Arm1', 'Arm2', 'Arm1'],
                                rewards=[20, 17, 25, 9],
                                learning_policy=LearningPolicy.UCB1(),
                                neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                context_history=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]],
                                contexts=[[2, 3, 1, 0]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        # self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["criterion"], "gini")
        self.assertIsNone(mab._imp.arm_to_tree[arm].__dict__["max_depth"])
        self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["min_samples_split"], 2)
        self.assertIsNone(mab._imp.arm_to_tree[arm].__dict__["max_leaf_nodes"])

    def test_tree_parameters(self):

        tree_parameters = {
            # "criterion": "entropy",
            "max_depth": 4,
            "min_samples_split": 2,
            "max_leaf_nodes": 10}

        arm, mab = self.predict(arms=['Arm1', 'Arm2'],
                                decisions=['Arm1', 'Arm1', 'Arm2', 'Arm1'],
                                rewards=[20, 17, 25, 9],
                                learning_policy=LearningPolicy.UCB1(),
                                neighborhood_policy=NeighborhoodPolicy.TreeBandit(tree_parameters=tree_parameters),
                                context_history=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]],
                                contexts=[[2, 3, 1, 0]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        # self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["criterion"], "entropy")
        self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["max_depth"], 4)
        self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["min_samples_split"], 2)
        self.assertEqual(mab._imp.arm_to_tree[arm].__dict__["max_leaf_nodes"], 10)

    def test_greedy0(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertListEqual(arms, [1, 1])

    def test_greedy0_single(self):
        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[2, 1, 1, 0, 0, 0, 4, 1, 1, 1],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 2, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 3)

    def test_greedy1(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[2, 1, 1, 0, 0, 0, 4, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=1.0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 2, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 4])

    def test_thompson(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_ucb(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(alpha=1),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 1])

    def test_greedy0_exps(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [0, 0, 1])
        self.assertListAlmostEqual(exps[1].values(), [0, 0, 1])

    def test_ucb_alpha0_exps(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(alpha=0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [0, 0, 1])
        self.assertListAlmostEqual(exps[1].values(), [0, 0, 1])

    def test_partial_fit_greedy0(self):
        arm, mab = self.predict(arms=['Arm1', 'Arm2'],
                                decisions=['Arm1', 'Arm1', 'Arm2', 'Arm1'],
                                rewards=[20, 17, 25, 9],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                context_history=[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]],
                                contexts=[[2, 3, 1, 0]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 'Arm2')

        decisions2 = ['Arm2']
        rewards2 = [30]
        contexts2 = [[2, 3, 0, 1]]
        mab.partial_fit(decisions2, rewards2, contexts2)

        values_1 = []
        for key in mab._imp.arm_to_leaf_to_rewards['Arm1'].keys():
            values_1.extend(mab._imp.arm_to_leaf_to_rewards['Arm1'][key])
        self.assertListEqual(sorted(values_1), [9, 17, 20])

        values_2 = []
        for key in mab._imp.arm_to_leaf_to_rewards['Arm2'].keys():
            values_2.extend(mab._imp.arm_to_leaf_to_rewards['Arm2'][key])
        self.assertListEqual(sorted(values_2), [25, 30])

    def test_partial_fit_thompson_thresholds(self):
        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
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

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        values_1 = []
        for key in mab._imp.arm_to_leaf_to_rewards[1].keys():
            values_1.extend(mab._imp.arm_to_leaf_to_rewards[1][key])
        self.assertListEqual(sorted(values_1), [0, 1, 1, 1])

        values_2 = []
        for key in mab._imp.arm_to_leaf_to_rewards[2].keys():
            values_2.extend(mab._imp.arm_to_leaf_to_rewards[2][key])
        self.assertListEqual(sorted(values_2), [0, 0, 0])

        values_3 = []
        for key in mab._imp.arm_to_leaf_to_rewards[3].keys():
            values_3.extend(mab._imp.arm_to_leaf_to_rewards[3][key])
        self.assertListEqual(sorted(values_3), [0, 1, 1, 1, 1, 1])

        values_4 = []
        for key in mab._imp.arm_to_leaf_to_rewards[4].keys():
            values_4.extend(mab._imp.arm_to_leaf_to_rewards[4][key])
        self.assertListEqual(sorted(values_4), [])

    def test_fit_twice_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
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

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        values_1 = []
        for key in mab._imp.arm_to_leaf_to_rewards[1].keys():
            values_1.extend(mab._imp.arm_to_leaf_to_rewards[1][key])
        self.assertListEqual(sorted(values_1), [1])

        values_2 = []
        for key in mab._imp.arm_to_leaf_to_rewards[2].keys():
            values_2.extend(mab._imp.arm_to_leaf_to_rewards[2][key])
        self.assertListEqual(sorted(values_2), [0])

        values_3 = []
        for key in mab._imp.arm_to_leaf_to_rewards[3].keys():
            values_3.extend(mab._imp.arm_to_leaf_to_rewards[3][key])
        self.assertListEqual(sorted(values_3), [1])

        values_4 = []
        for key in mab._imp.arm_to_leaf_to_rewards[4].keys():
            values_4.extend(mab._imp.arm_to_leaf_to_rewards[4][key])
        self.assertListEqual(sorted(values_4), [])

    def test_add_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
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
        self.assertTrue(5 in mab._imp.arm_to_tree.keys())
        self.assertTrue(5 in mab._imp.arm_to_leaf_to_rewards.keys())

    def test_add_arm_result_match(self):

        arms_1, mab = self.predict(arms=[1, 2, 4],
                                   decisions=[1, 1, 1, 2, 2],
                                   rewards=[0, 1, 1, 0, 0],
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                   neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                   context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]],
                                   contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                   seed=123456,
                                   num_run=1,
                                   is_predict=True)

        mab.add_arm(3)

        decisions2 = [3, 3, 3, 3, 3]
        rewards2 = [0, 0, 1, 1, 1]
        context_history2 = [[0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        arms_2 = mab._imp.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        self.assertListEqual(arms_1, [1, 1])
        self.assertListEqual(arms_1, arms_2)

    def test_series(self):
        arms = [0, 1]
        decisions = [1, 1, 1, 1, 1, 1]
        rewards = [0, 0, 0, 0, 0, 1]
        contexts = pd.DataFrame({'column1': [1, 2, 3, 3, 2, 1], 'column2': [2, 3, 1, 1, 2, 3]})

        mab = MAB(arms, LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.TreeBandit())
        mab.fit(decisions, rewards, contexts['column1'])
        result = mab.predict(pd.Series([1]))
        self.assertEqual(result, 1)

    def test_remove_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.arm_to_expectation)
        self.assertTrue(3 not in mab._imp.arm_to_tree)
        self.assertTrue(3 not in mab._imp.arm_to_leaf_to_rewards)
        self.assertTrue(3 not in mab._imp.lp.arms)

    def test_warm_start(self):
        arms_1, mab = self.predict(arms=[1, 2, 4],
                                   decisions=[1, 1, 1, 2, 2],
                                   rewards=[0, 1, 1, 0, 0],
                                   learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                   neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                   context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]],
                                   contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                   seed=123456,
                                   num_run=1,
                                   is_predict=True)

        mab.add_arm(3)

        arms_2 = mab._imp.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        self.assertListEqual(arms_1, [1, 1])
        self.assertListEqual(arms_1, arms_2)

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1], 4: [10, 10]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.predict_expectations([[1, 1, 1, 1, 1]]), {1: 1.0, 2: 0.0, 4: 0, 3: 0})
