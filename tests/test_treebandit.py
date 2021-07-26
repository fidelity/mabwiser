# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class TreeBanditTest(BaseTest):

    def test_usage_example(self):
        from mabwiser.mab import MAB, LearningPolicy
        list_of_arms = ['Arm1', 'Arm2']
        decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        rewards = [20, 17, 25, 9]
        contexts = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 1, 0], [3, 2, 1, 0]]
        mab = MAB(list_of_arms, LearningPolicy.UCB1(), NeighborhoodPolicy.TreeBandit())
        mab.fit(decisions, rewards, contexts)
        mab.predict([[3, 2, 0, 1]])

    def test_treebandit(self):
        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                    [0, 2, 1, 0, 0]])

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(),
                                neighborhood_policy=NeighborhoodPolicy.TreeBandit(),
                                context_history=context_history,
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[3, 3], [3, 3], [3, 3]])

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
        self.assertEqual(arm, 'Arm2')

    def test_treebandit_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(),
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

    def test_partial_fit(self):
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

        self.assertEqual(arm, 'Arm2')

        decisions = ['Arm2']
        rewards = [30]
        contexts = [[2, 3, 0, 1]]
        mab.partial_fit(decisions, rewards, contexts)

        values_1 = []
        for key in mab._imp.arm_to_rewards['Arm1'].keys():
            values_1.extend(mab._imp.arm_to_rewards['Arm1'][key])
        self.assertListEqual(sorted(values_1), [9, 17, 20])

        values_2 = []
        for key in mab._imp.arm_to_rewards['Arm2'].keys():
            values_2.extend(mab._imp.arm_to_rewards['Arm2'][key])
        self.assertListEqual(sorted(values_2), [25, 30])

    def test_tree_parameters(self):
        #TODO
        return

    def test_invalid_lp_linucb(self):
        #TODO
        return

    def test_invalid_lp_lints(self):
        #TODO
        return

    def test_all_context_free_lp(self):
        #TODO
        return