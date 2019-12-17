# -*- coding: utf-8 -*-

import datetime
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


#TODO what other tests are needed?
class LinTSTest(BaseTest):

    def test_alpha0_0001(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=0.0001),
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

    def test_alpha0_0001_expectations(self):

        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                 learning_policy=LearningPolicy.LinTS(alpha=0.0001),
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

    def test_alpha1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
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

        self.assertEqual(len(arm), 2)
        self.assertEqual(arm, [3, 1])

    def test_alpha1_expectations(self):

        exps, mab = self.predict(arms=[1, 2, 3],
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

        self.assertListAlmostEqual(exps[0].values(), [-0.8290910888393794, -2.484684983207152, 0.19318917358066356])
        self.assertListAlmostEqual(exps[1].values(), [0.718081042085166, -0.04901496551413326, -0.4584390221403405])

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
        self.assertEqual(arm, [[3, 1], [3, 3], [1, 3]])

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
        self.assertEqual(arm, [[3, 1], [3, 3], [1, 3]])

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
        self.assertEqual(arm, [[3, 1], [3, 3], [1, 3]])

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
        self.assertEqual(arm, [[2, 1], [1, 1], [1, 1], [2, 1]])

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
        self.assertEqual(arm, [[1, 3], [2, 2], [3, 2], [3, 1]])

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
        self.assertEqual(arm, [[4, 4], [1, 4], [4, 4], [1, 4]])

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
        self.assertEqual(arm, [[1, 4], [4, 4], [4, 4], [4, 4]])

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
        self.assertEqual(arm, [['one', 'two'], ['two', 'two'], ['two', 'one'], ['two', 'three']])

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
        self.assertEqual(arm, [['three', 'one'], ['three', 'one'], ['two', 'one'], ['three', 'one']])

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
                                seed=7,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [[b, b],[b, b], [c, b], [b, b]])

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

        self.assertTrue(4 in mab._imp.unused)

        self.assertListAlmostEqual(exps[0].values(), [-0.8290910888393794, -2.484684983207152,
                                                      0.19318917358066356, 7.100760971201706])
        self.assertListAlmostEqual(exps[1].values(), [0.00301615122968224, -0.3578976880690734,
                                                      2.066646242982663, 0.46721497144397417])

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

        self.assertTrue(4 in mab._imp.unused)
        self.assertEqual(arms, [4, 3])

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

        self.assertTrue(4 in mab._imp.unused)
        self.assertListAlmostEqual(exps[0].values(), [1.6545684352344563, 1.8845226500371124,
                                                      -0.3600696894227234, 0.8920650756566723])
        self.assertListAlmostEqual(exps[1].values(), [0.7128765844473357, -0.24246325711739597,
                                                      0.9121315140497883, -1.022929456258629])

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

        self.assertTrue(4 in mab._imp.unused)
        self.assertEqual(arms, [4, 2])

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

        self.assertEqual(arm, [4, 3])
        self.assertTrue(4 in mab._imp.unused)

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

        self.assertTrue(2 in mab._imp.unused)

        b_1 = mab._imp.arm_to_model[1].beta
        self.assertEqual(b_1[0], 0)

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

        self.assertEqual(arm, [4, 3])
        self.assertTrue(4 in mab._imp.unused)

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

        self.assertFalse(mab._imp.unused)
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
                                learning_policy=LearningPolicy.LinTS(alpha=0),
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
        self.assertEqual(arm, [[3, 2], [3, 2], [3, 2]])

    def test_alpha0_nearest5(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.LinTS(alpha=0),
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
        self.assertEqual(arm, [[3, 3], [3, 3], [3, 3]])
