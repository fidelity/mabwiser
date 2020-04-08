# -*- coding: utf-8 -*-

import datetime
import math

import numpy as np
import pandas as pd

from mabwiser.mab import LearningPolicy
from tests.test_base import BaseTest


class SoftmaxTest(BaseTest):

    def test_tau0(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=0.00001),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [1, 1, 1, 1])

    def test_tau0_expectations(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=0.00001),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        self.assertDictEqual(arm, {1: 1.0, 2: 0.0, 3: 0.0})

    def test_tau1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3, 1])

    def test_tau1_expectations(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        self.assertDictEqual(arm, {1: 0.4083425853583662, 2: 0.20965007375301267, 3: 0.3820073408886212})

    def test_tau1_np(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                                rewards=np.asarray([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3, 1])

    def test_tau1_df(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3], 'rewards': [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=df['rewards'],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3, 1])

    def test_tau1_df_list(self):

        df = pd.DataFrame({'decisions': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3], 'rewards': [0, 1, 1, 0, 0, 0, 0, 1, 1, 1]})

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=df['decisions'],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3, 1])

    def test_softmax_t1(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=0.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3])

    def test_softmax_t2(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=0.5),
                                seed=71,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [1, 1, 3, 1])

    def test_softmax_t3(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [4, 4, 4, 4])

    def test_softmax_t4(self):

        arm, mab = self.predict(arms=[1, 2, 4],
                                decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                learning_policy=LearningPolicy.Softmax(tau=2),
                                seed=23,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [4, 4, 4, 4])

    def test_softmax_t5(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                rewards=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                learning_policy=LearningPolicy.Softmax(tau=1.5),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, ['one', 'three', 'one', 'three'])

    def test_softmax_t6(self):

        arm, mab = self.predict(arms=['one', 'two', 'three'],
                                decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=17,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, ['two', 'three', 'one', 'one'])

    def test_softmax_t7(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.Softmax(tau=0.25),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, ['b', 'c', 'b', 'c', 'b'])

    def test_softmax_t8(self):

        arm, mab = self.predict(arms=['a', 'b', 'c'],
                                decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.Softmax(tau=0.5),
                                seed=9,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, ['c', 'c', 'c', 'c', 'c'])

    def test_softmax_t9(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a],
                                rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                learning_policy=LearningPolicy.Softmax(tau=1.25),
                                seed=123456,
                                num_run=4,
                                is_predict=True)

        self.assertEqual(arm, [c, c, c, c])

    def test_softmax_t10(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arm, mab = self.predict(arms=[a, b, c],
                                decisions=[a, b, c, a, b, c, a, b, c, a, b, b],
                                rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7],
                                learning_policy=LearningPolicy.Softmax(tau=0.33),
                                seed=7,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [b, b, b, b, b])

    def test2_unused_arm(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=20,
                                is_predict=True)

        self.assertTrue(4 in mab._imp.arm_to_expectation.keys())
        self.assertEqual(arm[13], 4)

        e_x = mab._imp.arm_to_exponent[4]
        prob = mab._imp.arm_to_expectation[4]

        self.assertTrue(math.isclose(e_x, 0.513, abs_tol=0.001))
        self.assertTrue(math.isclose(prob, 0.173, abs_tol=0.001))

    def test_fit_twice(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        mean = mab._imp.arm_to_mean[1]
        ex = mab._imp.arm_to_exponent[1]
        prob = arm[1]

        self.assertTrue(math.isclose(mean, 0.667, abs_tol=0.001))
        self.assertTrue(math.isclose(ex, 1))
        self.assertTrue(math.isclose(prob, 0.338, abs_tol=0.001))

        self.assertTrue(4 in arm.keys())
        mean1 = mab._imp.arm_to_mean[4]
        ex1 = mab._imp.arm_to_exponent[4]
        prob1 = arm[4]

        self.assertEqual(mean1, 0)
        self.assertTrue(math.isclose(ex1, 0.513, abs_tol=0.001))
        self.assertTrue(math.isclose(prob1, 0.173, abs_tol=0.001))

        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        mab.fit(decisions2, rewards2)

        mean2 = mab._imp.arm_to_mean[1]
        ex2 = mab._imp.arm_to_exponent[1]
        prob2 = mab._imp.arm_to_expectation[1]
        mean3 = mab._imp.arm_to_mean[4]
        ex3 = mab._imp.arm_to_exponent[4]
        prob3 = mab._imp.arm_to_expectation[4]

        self.assertEqual(mean2, 0)
        self.assertTrue(math.isclose(ex2, 0.368, abs_tol=0.001))
        self.assertTrue(math.isclose(prob2, 0.134, abs_tol=0.001))
        self.assertEqual(mean3, 1)
        self.assertEqual(ex3, 1.0)
        self.assertTrue(math.isclose(prob3, 0.366, abs_tol=0.001))

    def test_partial_fit(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        mean = mab._imp.arm_to_mean[1]
        ex = mab._imp.arm_to_exponent[1]
        prob = arm[1]

        self.assertTrue(math.isclose(mean, 0.667, abs_tol=0.001))
        self.assertTrue(math.isclose(ex, 1.0, abs_tol=0.001))
        self.assertTrue(math.isclose(prob, 0.338, abs_tol=0.001))

        self.assertTrue(4 in arm.keys())
        mean1 = mab._imp.arm_to_mean[4]
        ex1 = mab._imp.arm_to_exponent[4]
        prob1 = arm[4]

        self.assertEqual(mean1, 0)
        self.assertTrue(math.isclose(ex1, 0.513, abs_tol=0.001))
        self.assertTrue(math.isclose(prob1, 0.173, abs_tol=0.001))

        decisions2 = [1, 3, 4]
        rewards2 = [0, 1, 1]
        mab.partial_fit(decisions2, rewards2)

        mean2 = mab._imp.arm_to_mean[1]
        ex2 = mab._imp.arm_to_exponent[1]
        prob2 = mab._imp.arm_to_expectation[1]
        mean3 = mab._imp.arm_to_mean[4]
        ex3 = mab._imp.arm_to_exponent[4]
        prob3 = mab._imp.arm_to_expectation[4]

        self.assertEqual(mean2, 0.5)
        self.assertTrue(math.isclose(ex2, 0.607, abs_tol=0.001))
        self.assertTrue(math.isclose(prob2, 0.225, abs_tol=0.001))
        self.assertEqual(mean3, 1)
        self.assertEqual(ex3, 1.0)
        self.assertTrue(math.isclose(prob3, 0.372, abs_tol=0.001))

    def test_add_arm(self):

        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        mab.add_arm(5)
        self.assertTrue(5 in mab.arms)
        self.assertTrue(5 in mab._imp.arms)
        self.assertTrue(5 in mab._imp.arm_to_expectation.keys())
        self.assertTrue(mab._imp.arm_to_mean[5] == 0)
        self.assertTrue(mab._imp.arm_to_expectation[4] == mab._imp.arm_to_expectation[5])
