import datetime
import math
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class LinGreedyTest(BaseTest):

    def test_epsilon_zero(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[3, 3], [3, 3], [3, 3]])

    def test_epsilon_zero_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(epsilon=0),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [-0.018378378378378413, 0.0, 0.9966292134831471])
        self.assertListAlmostEqual(exps[1].values(), [0.14054054054054055, 0.0, 0.43258426966292074])

    def test_epsilon_zero_vs_linucb(self):
        arm_lingreedy, mab_lingreedy = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0, l2_lambda=0.87),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        arm_linucb, mab_linucb = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinUCB(alpha=0, l2_lambda=0.87),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(arm_lingreedy, arm_linucb)

    def test_epsilon_zero_vs_linucb_expectations(self):
        exps_lingreedy, mab_lingreedy = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0, l2_lambda=0.29),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=False)

        exps_linucb, mab_linucb = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinUCB(alpha=0, l2_lambda=0.29),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=False)

        self.assertEqual(exps_lingreedy, exps_linucb)

    def test_epsilon_one(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[2, 1], [1, 2], [3, 3]])

    def test_epsilon_one_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(epsilon=0),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [-0.018378378378378413, 0.0, 0.9966292134831471])
        self.assertListAlmostEqual(exps[1].values(), [0.14054054054054055, 0.0, 0.43258426966292074])

    def test_np(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                                rewards=np.asarray([0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[2, 1], [1, 2], [3, 3]])

    def test_df(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                           'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=df['rewards'],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=pd.DataFrame([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                              [0, 2, 1, 0, 0]]),
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[2, 1], [1, 2], [3, 3]])

    def test_df_list(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                           'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[2, 1], [1, 2], [3, 3]])

    def test_lingreedy_t1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.25),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[2, 1], [2, 1], [2, 1], [2, 1]])

    def test_lingreedy_t2(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[3, 1], [2, 1], [1, 3], [2, 3]])

    def test_lingreedy_t3(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4, 1],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.75),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5],
                                                 [1, 2, 1, 1, 3], [0, 2, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[4, 1], [4, 2], [4, 4], [4, 1]])

    def test_lingreedy_t4(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4, 1],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5],
                                                 [1, 2, 1, 1, 3], [0, 2, 1, 0, 0], [0, 2, 2, 3, 5], [1, 3, 1, 1, 1]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=23,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[2, 2], [2, 1], [1, 1], [4, 4]])

    def test_lingreedy_t5(self):

        arm, mab = self.predict(arms=["one", "two", "three"],
                                decisions=["one", "one", "one", "three", "two", "two", "three", "one", "three", "one"],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [["three", "one"], ["two", "one"], ["one", "three"], ["two", "three"]])

    def test_lingreedy_t6(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two',
                                           'one'],
                                rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.75),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 4, 3, 5]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=17,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [['three', 'two'], ['three', 'two'], ['two', 'one'], ['two', 'one']])

    def test_lingreedy_t7(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'a'],
                                rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 4, 3, 5]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=17,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [['c', 'b'], ['c', 'b'], ['c', 'b'], ['c', 'b']])

    def test_lingreedy_t8(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=9,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [['c', 'c'], ['c', 'c'], ['c', 'c'], ['c', 'c']])

    def test_lingreedy_t9(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a],
                                rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.25),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[c, a], [c, c], [c, c], [c, a]])

    def test_lingreedy_t10(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a, b, b, a],
                                rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=7,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[c, b], [b, b], [b, b], [a, c]])

    def test_unused_arm(self):

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [0.004587155963302836, 0.0, 0.6208530805687187, 0.0])
        self.assertListAlmostEqual(exps[1].values(),
                                   [0.9026245819618994, 0.08302023258745872, 0.14184572228349002, 0.625881702106679])

    def test_unused_arm2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [3, 1])

    def test_unused_arm_scaled(self):

        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                    [0, 2, 1, 0, 0]], dtype='float64')

        scaler = StandardScaler()
        scaled_contexts = scaler.fit_transform(context_history)
        scaled_predict = scaler.transform(np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], dtype='float64'))

        exp, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=scaled_contexts,
                                contexts=scaled_predict,
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        self.assertListAlmostEqual(exp[0].values(), [-0.07721690708221138, 0.0, 0.1366841348928508, 0.0])
        self.assertListAlmostEqual(exp[1].values(),
                                   [0.9026245819618994, 0.08302023258745872, 0.14184572228349002, 0.625881702106679])

    def test_unused_arm_scaled2(self):

        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5],
                                    [1, 3, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 4, 3, 5], [0, 1, 2, 4, 5],
                                    [1, 2, 1, 1, 3], [0, 2, 1, 0, 0]], dtype='float64')

        scaler = StandardScaler()
        scaled_contexts = scaler.fit_transform(context_history)
        scaled_predict = scaler.transform(np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                 context_history=scaled_contexts,
                                 contexts=scaled_predict,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [3, 1])

    def test_fit_twice(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, [3, 1])

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertTrue(math.isclose(-0.0825688, b_1[0], abs_tol=0.00001))

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(0.023696, b_3[0], abs_tol=0.00001))

        self.assertTrue(4 in mab._imp.arm_to_model.keys())

        # # Fit again
        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        context_history2 = [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertEqual(b_1[0], 0)

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(b_3[0], 0.16667, abs_tol=0.00001))

        b_4 = mab._imp.arm_to_model[4].beta
        self.assertEqual(b_4[0], 0)

    def test_fit_twice_scale(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        self.assertEqual(arm, [3, 1])
        self.assertAlmostEqual(mab._imp.arm_to_model[1].beta[0], -0.1520794283674759)
        self.assertAlmostEqual(mab._imp.arm_to_model[2].beta[0], 0)
        self.assertAlmostEqual(mab._imp.arm_to_model[3].beta[0], -0.008110550702115856)
        self.assertAlmostEqual(mab._imp.arm_to_model[4].beta[0], 0)

        # Fit again
        mab.fit(decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                          [0, 2, 1, 0, 0]])
        self.assertEqual(arm, [3, 1])
        self.assertAlmostEqual(mab._imp.arm_to_model[1].beta[0], -0.1520794283674759)
        self.assertAlmostEqual(mab._imp.arm_to_model[2].beta[0], 0)
        self.assertAlmostEqual(mab._imp.arm_to_model[3].beta[0], -0.008110550702115856)
        self.assertAlmostEqual(mab._imp.arm_to_model[4].beta[0], 0)

    def test_add_arm_scale(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        mab.add_arm(4)
        mab.partial_fit(decisions=[1, 1, 4, 4],
                        rewards=[0, 1, 1, 1],
                        contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 2, 2, 3, 5]])
        self.assertEqual(mab._imp.arm_to_model[1].scaler.n_samples_seen_, 5)
        self.assertEqual(mab._imp.arm_to_model[2].scaler.n_samples_seen_, 2)
        self.assertEqual(mab._imp.arm_to_model[3].scaler.n_samples_seen_, 5)
        self.assertEqual(mab._imp.arm_to_model[4].scaler.n_samples_seen_, 2)
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[1].scaler.mean_), [0.4, 0.8, 1.4, 1.6, 2.4])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[2].scaler.mean_), [0.5, 2.5, 1.5, 2.0, 3.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[3].scaler.mean_), [0.2, 1.2, 1.6, 1.6, 2.6])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[4].scaler.mean_), [0.0, 1.0, 1.5, 1.5, 2.5])

    def test_partial_fit(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, [3, 1])

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertTrue(math.isclose(-0.0825688, b_1[0], abs_tol=0.00001))

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(0.023696, b_3[0], abs_tol=0.00001))

        self.assertTrue(4 in mab._imp.arm_to_model.keys())

        # Fit again
        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        context_history2 = [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertTrue(math.isclose(-0.05142857, b_1[0], abs_tol=0.00001))

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(b_3[0], 0.22099152, abs_tol=0.00001))

        b_4 = mab._imp.arm_to_model[4].beta
        self.assertEqual(b_4[0], 0)

    def test_alpha0_radius1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                neighborhood_policy=NeighborhoodPolicy.Radius(radius=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[3, 2], [3, 1], [3, 1]])

    def test_alpha0_nearest5(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                neighborhood_policy=NeighborhoodPolicy.KNearest(k=5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[3, 2], [3, 3], [3, 3]])

    def test_scaler_fit(self):
        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        self.assertEqual(mab._imp.arm_to_model[1].scaler.n_samples_seen_, 3)
        self.assertEqual(mab._imp.arm_to_model[2].scaler.n_samples_seen_, 2)
        self.assertEqual(mab._imp.arm_to_model[3].scaler.n_samples_seen_, 5)
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[1].scaler.mean_), [1/3, 2/3,  4/3, 4/3, 2.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[2].scaler.mean_), [0.5, 2.5, 1.5, 2.0, 3.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[3].scaler.mean_), [0.2, 1.2, 1.6, 1.6, 2.6])

    def test_scaler_fit_twice(self):
        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        self.assertEqual(mab._imp.arm_to_model[1].scaler.n_samples_seen_, 3)
        self.assertEqual(mab._imp.arm_to_model[2].scaler.n_samples_seen_, 2)
        self.assertEqual(mab._imp.arm_to_model[3].scaler.n_samples_seen_, 5)
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[1].scaler.mean_), [1/3, 2/3,  4/3, 4/3, 2.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[2].scaler.mean_), [0.5, 2.5, 1.5, 2.0, 3.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[3].scaler.mean_), [0.2, 1.2, 1.6, 1.6, 2.6])

        mab.fit(decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                          [0, 2, 1, 0, 0]])
        self.assertEqual(mab._imp.arm_to_model[1].scaler.n_samples_seen_, 3)
        self.assertEqual(mab._imp.arm_to_model[2].scaler.n_samples_seen_, 2)
        self.assertEqual(mab._imp.arm_to_model[3].scaler.n_samples_seen_, 5)
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[1].scaler.mean_), [1/3, 2/3,  4/3, 4/3, 2.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[2].scaler.mean_), [0.5, 2.5, 1.5, 2.0, 3.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[3].scaler.mean_), [0.2, 1.2, 1.6, 1.6, 2.6])

    def test_scaler_partial_fit(self):
        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        mab.partial_fit(decisions=[3, 3, 3, 2, 2, 1, 1, 1, 1, 1],
                        rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                        contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                  [0, 2, 1, 0, 0]])
        self.assertEqual(mab._imp.arm_to_model[1].scaler.n_samples_seen_, 8)
        self.assertEqual(mab._imp.arm_to_model[2].scaler.n_samples_seen_, 4)
        self.assertEqual(mab._imp.arm_to_model[3].scaler.n_samples_seen_, 8)
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[1].scaler.mean_), [0.25, 1.0, 1.5, 1.5, 2.375])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[2].scaler.mean_), [0.5, 2.5, 1.5, 2.0, 3.0])
        self.assertListAlmostEqual(list(mab._imp.arm_to_model[3].scaler.mean_), [0.25, 1.0, 1.5, 1.5, 2.375])

    def test_scaler_predictions(self):

        arms = [1, 2, 3]
        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                    [0, 2, 1, 0, 0]], dtype='float64')

        contexts = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
        decisions = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
        rewards = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])

        arm_to_scaler = {}
        for arm in arms:
            scaler = StandardScaler()
            df = context_history[decisions == arm]
            scaler.fit(np.asarray(df, dtype='float64'))
            arm_to_scaler[arm] = deepcopy(scaler)

        exp, mab = self.predict(arms=arms,
                                decisions=decisions,
                                rewards=rewards,
                                learning_policy=LearningPolicy.LinGreedy(scale=True),
                                context_history=context_history,
                                contexts=contexts,
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        for arm in arms:

            context_history_arm = context_history[decisions == arm]
            context_history_scaled = arm_to_scaler[arm].transform(np.asarray(context_history_arm, dtype='float64'))

            contexts_scaled = arm_to_scaler[arm].transform(np.asarray(contexts, dtype='float64'))

            exp_check, mab = self.predict(arms=arms,
                                          decisions=decisions[decisions == arm],
                                          rewards=rewards[decisions == arm],
                                          learning_policy=LearningPolicy.LinGreedy(),
                                          context_history=context_history_scaled,
                                          contexts=contexts_scaled,
                                          seed=123456,
                                          num_run=1,
                                          is_predict=False)

            for i in range(len(contexts)):
                self.assertEqual(exp[i][arm], exp_check[i][arm])

    def test_unused_arm_scale(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinGreedy(l2_lambda=1, scale=True),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [3, 1])

    def test_add_arm(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.75),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)
        mab.add_arm(4)
        self.assertTrue(4 in mab.arms)
        self.assertTrue(4 in mab._imp.arms)
        self.assertTrue(4 in mab._imp.arm_to_expectation.keys())
        self.assertTrue(mab._imp.arm_to_model[4] is not None)

    def test_remove_arm(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinGreedy(epsilon=0.75),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.arm_to_expectation)
        self.assertTrue(3 not in mab._imp.arm_to_model)

    def test_warm_start(self):
        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 1, 2, 2, 2, 1, 2, 1],
                              rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                              learning_policy=LearningPolicy.LinGreedy(epsilon=0.25),
                              context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                               [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                               [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                               [0, 2, 1, 0, 0]],
                              contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                              seed=123456,
                              num_run=4,
                              is_predict=True)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.0, 2: 0.0, 3: 0.0})
        self.assertListAlmostEqual(mab._imp.arm_to_model[1].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0, 0, 0, 0, 0])

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
