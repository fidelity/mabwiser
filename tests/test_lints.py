# -*- coding: utf-8 -*-

import datetime
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy
from tests.test_base import BaseTest


class LinTSTest(BaseTest):

    def test_alpha0_0001(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=0.0001, scale=True),
                                context_history=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                         [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                         [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                         [0, 2, 1, 0, 0]]),
                                contexts=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]]),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[2, 2], [2, 2], [2, 3]])

    def test_alpha0_0001_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=0.0001, scale=True),
                                 context_history=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                          [0, 2, 1, 0, 0]]),
                                 contexts=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]]),
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(),
                                   [-0.23448413444794053, 5.8260939841235566e-05, -2.2378841145535292e-05])
        self.assertListAlmostEqual(exps[1].values(),
                                   [-0.19261783297618085, 8.502597859929583e-05, -0.0002446465621757603])

    def test_alpha1(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                         [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                         [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                         [0, 2, 1, 0, 0]]),
                                contexts=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]]),
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        self.assertEqual(len(arm), 2)
        self.assertEqual(arm, [3, 3])

    def test_alpha1_expectations(self):
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 context_history=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                           [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                           [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                           [0, 2, 1, 0, 0]]),
                                 contexts=np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]]),
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)
        self.assertListAlmostEqual(exps[0].values(), [0.35882688729202394, 0.4531149510422985, 0.5724694641123476])
        self.assertListAlmostEqual(exps[1].values(), [1.834757741062738, 1.8649301985477622, 2.3052939555570764])

    def test_np(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                                rewards=np.asarray([0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
                                learning_policy=LearningPolicy.LinTS(alpha=1),
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

    def test_df(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                           'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=df['rewards'],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=pd.DataFrame([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                              [0, 2, 1, 0, 0]]),
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [[3, 3], [3, 3], [3, 3]])

    def test_df_list(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                           'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
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

    def test_lints_t1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=0.24),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[2, 1], [2, 1], [1, 1], [1, 1]])

    def test_lints_t2(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1.5),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[1, 3], [2, 1], [1, 3], [2, 1]])

    def test_lints_t3(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4, 1],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1.25),
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
        self.assertEqual(arm, [[4, 4], [4, 4], [1, 4], [1, 4]])

    def test_lints_t4(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4, 1],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=2),
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
        self.assertEqual(arm, [[4, 4], [4, 4], [4, 4], [2, 4]])

    def test_lints_t5(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                rewards=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=23,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [['one', 'one'], ['two', 'two'], ['one', 'three'], ['two', 'three']])

    def test_lints_t6(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two',
                                           'one'],
                                rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1.25),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 4, 3, 5]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=17,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [['three', 'one'], ['three', 'one'], ['three', 'one'], ['three', 'one']])

    def test_lints_t7(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'a'],
                                rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, ['b', 'b', 'b', 'b'])

    def test_lints_t8(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.LinTS(alpha=0.5),
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

    def test_lints_t9(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a],
                                rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.LinTS(alpha=0.25),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[c, c], [c, c], [c, c], [c, c]])

    def test_lints_t10(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a, b, b, a],
                                rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0], [0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[b, b], [b, b], [b, b], [c, b]])

    def test_unused_arm_scale(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1, scale=True),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4])

    def test_unused_arm(self):

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [0.35882688729202394, 0.4531149510422985,
                                                      0.5724694641123476, 6.996743596391253])
        self.assertListAlmostEqual(exps[1].values(), [1.834757741062738, 1.8649301985477622,
                                                      2.3052939555570764, 3.197481957191733])

    def test_unused_arm2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4])

    def test_unused_arm_scaled(self):

        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]], dtype='float64')
        scaler = StandardScaler()
        scaled_contexts = scaler.fit_transform(context_history)
        scaled_predict = scaler.transform(np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], dtype='float64'))

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 context_history=scaled_contexts,
                                 contexts=scaled_predict,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertListAlmostEqual(exps[0].values(), [0.3378166210104375, 0.6254259327060168,
                                                      0.12963723386162468, 1.1684828880341767])
        self.assertListAlmostEqual(exps[1].values(), [0.21582194096636984, 0.10707407671505453,
                                                      0.3819545461857511, -1.431880645181654])

    def test_unused_arm_scaled2(self):

        context_history = np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]], dtype='float64')
        scaler = StandardScaler()
        scaled_contexts = scaler.fit_transform(context_history)
        scaled_predict = scaler.transform(np.array([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 context_history=scaled_contexts,
                                 contexts=scaled_predict,
                                 seed=7,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [3, 4])

    def test_fit_twice(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, [4, 4])

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertTrue(math.isclose(-0.0825688, b_1[0], abs_tol=0.00001))

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(0.023696, b_3[0], abs_tol=0.00001))

        self.assertTrue(4 in mab._imp.arm_to_model.keys())

        # Fit again
        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        context_history2 = [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertEqual(b_1[0], 0)

        b_2 = mab._imp.arm_to_model[2].beta
        self.assertEqual(b_2[0], 0)

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(b_3[0], 0.16667, abs_tol=0.00001))

        b_4 = mab._imp.arm_to_model[4].beta
        self.assertEqual(b_4[0], 0)

    def test_partial_fit(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=1),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, [4, 4])

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
        b_2 = mab._imp.arm_to_model[2].beta
        self.assertEqual(b_2[0], 0)

        b_3 = mab._imp.arm_to_model[3].beta
        self.assertTrue(math.isclose(b_3[0], 0.22099152, abs_tol=0.00001))

        b_4 = mab._imp.arm_to_model[4].beta
        self.assertEqual(b_4[0], 0)

    def test_add_arm(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3, 1],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=0.24),
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
                                learning_policy=LearningPolicy.LinTS(alpha=0.24),
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
                              learning_policy=LearningPolicy.LinTS(alpha=0.24),
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
        self.assertListAlmostEqual(mab._imp.arm_to_model[1].beta, [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0, 0, 0, 0, 0])

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])

    def test_double_warm_start(self):
        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 1, 2, 2, 2, 1, 2, 1],
                              rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                              learning_policy=LearningPolicy.LinTS(alpha=0.24),
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
        self.assertListAlmostEqual(mab._imp.arm_to_model[1].beta, [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0, 0, 0, 0, 0])

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])

        # Warm start again, #3 shouldn't change even though it's closer to #2 now
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
