# -*- coding: utf-8 -*-

import datetime
import math

import numpy as np
import pandas as pd

from mabwiser.mab import LearningPolicy
from mabwiser.ucb import _UCB1
from tests.test_base import BaseTest


class UCBTest(BaseTest):

    def test_alpha0(self):
        
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=0),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [3, 3, 3])

    def test_alpha0_expectations(self):
        
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=0),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        self.assertDictEqual(arm, {1: 0.0, 2: 0.0, 3: 1.0})

    def test_alpha1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [1, 1, 1])

    def test_alpha1_expectations(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        self.assertDictEqual(arm, {1:  1.5723073962832794, 2: 1.5174271293851465, 3: 1.5597051824376162})

    def test_np(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                                rewards=np.asarray([0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [1, 1, 1])

    def test_df(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3], 'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=df['rewards'],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [1, 1, 1])

    def test_df_list(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3], 'rewards': [0, 0, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=3,
                                is_predict=True)

        self.assertEqual(len(arm), 3)
        self.assertEqual(arm, [1, 1, 1])

    def test_ucb_t1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=0.24),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [1, 1, 1, 1])

    def test_ucb_t2(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1.5),
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [2, 2, 2, 2])

    def test_ucb_t3(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [4, 4, 4, 4])

    def test_ucb_t4(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                learning_policy=LearningPolicy.UCB1(alpha=2),
                                seed=23,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [4, 4, 4, 4])

    def test_ucb_t5(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                rewards=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=23,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, ['three', 'three', 'three', 'three'])

    def test_ucb_t6(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4],
                                learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                seed=17,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, ['three', 'three', 'three', 'three'])

    def test_ucb_t7(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, ['b', 'b', 'b', 'b'])

    def test_ucb_t8(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.UCB1(alpha=0.5),
                                seed=9,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, ['c', 'c', 'c', 'c'])

    def test_ucb_t9(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a],
                                rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.UCB1(alpha=0.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [c, c, c, c])

    def test_ucb_t10(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a, b, b],
                                rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=7,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(len(arm), 4)
        self.assertEqual(arm, [b, b, b, b])

    def test_unused_arm(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertTrue(len(mab._imp.arm_to_expectation), 4)

    def test_fit_twice(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertTrue(len(mab._imp.arm_to_expectation), 4)

        mean = mab._imp.arm_to_mean[1]
        ci = mab._imp.arm_to_expectation[1]

        self.assertAlmostEqual(0.3333333333333333, mean)
        self.assertAlmostEqual(1.5723073962832794, ci)

        mean1 = mab._imp.arm_to_mean[4]
        ci1 = mab._imp.arm_to_expectation[4]
        self.assertEqual(mean1, 0)
        self.assertEqual(ci1, 0)

        # Fit again
        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        mab.fit(decisions2, rewards2)

        mean2 = mab._imp.arm_to_mean[1]
        ci2 = mab._imp.arm_to_expectation[1]
        mean3 = mab._imp.arm_to_mean[4]
        ci3 = mab._imp.arm_to_expectation[4]

        self.assertEqual(mean2, 0)
        self.assertAlmostEqual(0, mean2)
        self.assertAlmostEqual(1.4823038073675112, ci2)
        self.assertEqual(mean3, 1)
        self.assertAlmostEqual(2.4823038073675114, ci3)

    def test_partial_fit(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.UCB1(alpha=1),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertTrue(len(mab._imp.arm_to_expectation), 4)

        mean = mab._imp.arm_to_mean[1]
        ci = mab._imp.arm_to_expectation[1]
        mean1 = mab._imp.arm_to_mean[2]
        ci1 = mab._imp.arm_to_expectation[2]
        self.assertAlmostEqual(0.3333333333333333, mean)
        self.assertAlmostEqual(1.5723073962832794, ci)
        self.assertEqual(mean1, 0)
        self.assertAlmostEqual(ci1, 1.5174271293851465)

        mean2 = mab._imp.arm_to_mean[4]
        ci2 = mab._imp.arm_to_expectation[4]
        self.assertEqual(mean2, 0)
        self.assertEqual(ci2, 0)

        # Fit again
        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        mab.partial_fit(decisions2, rewards2)

        mean3 = mab._imp.arm_to_mean[1]
        ci3 = mab._imp.arm_to_expectation[1]
        mean4 = mab._imp.arm_to_mean[4]
        ci4 = mab._imp.arm_to_expectation[4]
        mean5 = mab._imp.arm_to_mean[2]
        ci5 = mab._imp.arm_to_expectation[2]

        self.assertEqual(mean3, 0.25)
        self.assertAlmostEqual(1.3824639856219572, ci3)
        self.assertEqual(mean4, 1)
        self.assertAlmostEqual(3.2649279712439143, ci4)
        self.assertEqual(mean5, 0)
        self.assertAlmostEqual(ci5, 1.6015459273656616)

    def test_add_arm(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 2, 1, 1, 2],
                                rewards=[10, 4, 3, 5, 6],
                                learning_policy=LearningPolicy.UCB1(1.0),
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        mab.add_arm(4)
        self.assertTrue(4 in mab.arms)
        self.assertTrue(4 in mab._imp.arms)
        self.assertTrue(mab._imp.arm_to_expectation[4] == 0)
        self.assertTrue(mab._imp.arm_to_mean[4] == 0)

    def test_confidence(self):

        # parameters
        mean = 20
        arm_count = 150
        total_count = 500

        alpha = 1
        cb = _UCB1._get_ucb(mean, alpha, total_count, arm_count)
        self.assertAlmostEqual(cb, 20.287856633260894)

        alpha = 0.25
        cb = _UCB1._get_ucb(mean, alpha, total_count, arm_count)
        self.assertAlmostEqual(cb, 20.07196415831522)

        alpha = 3.33
        cb = _UCB1._get_ucb(mean, alpha, total_count, arm_count)
        self.assertAlmostEqual(cb, 20.95856258875877)

    def test_remove_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(1.0),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.arm_to_sum)
        self.assertTrue(3 not in mab._imp.arm_to_count)
        self.assertTrue(3 not in mab._imp.arm_to_mean)

    def test_warm_start(self):

        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.UCB1(1.0),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_mean, {1: 0.5, 2: 0.0, 3: 0.0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_mean, {1: 0.5, 2: 0.0, 3: 0.5})

    def test_double_warm_start(self):

        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.UCB1(1.0),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_mean, {1: 0.5, 2: 0.0, 3: 0.0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_mean, {1: 0.5, 2: 0.0, 3: 0.5})

        # Warm start again, #3 is closest to #2 but shouldn't get warm started again
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_mean, {1: 0.5, 2: 0.0, 3: 0.5})

    def test_ucb_contexts(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(),
                                 contexts=[[]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(),
                                 contexts=[[1, 2, 3]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
