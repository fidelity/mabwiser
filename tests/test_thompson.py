# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from mabwiser.mab import LearningPolicy
from tests.test_base import BaseTest


class ThompsonTest(BaseTest):

    def test_thompson(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.ThompsonSampling(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

    def test_thompson_boolean_reward(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[False, True, True, False, False, False, False, True, True, True],
                                learning_policy=LearningPolicy.ThompsonSampling(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

    def test_thompson_convert_to_binary(self):

        dec_to_threshold = {1: 7, 2: 12, 3: 0}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 2, 1, 1, 2],
                                rewards=[10, 4, 3, 5, 6],
                                learning_policy=
                                LearningPolicy.ThompsonSampling(binarize),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(mab._imp.arm_to_success_count[1], 2)
        self.assertEqual(mab._imp.arm_to_success_count[2], 1)
        self.assertEqual(mab._imp.arm_to_success_count[3], 1)

        self.assertEqual(mab._imp.arm_to_fail_count[1], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[2], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[3], 1)

        self.assertEqual(arm, 3)

    def test_thompson_multiple(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=5,
                                 is_predict=True)

        self.assertEqual(arms, [1, 3, 3, 1, 1])

    def test_thompson_unused_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=5,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4, 4, 1, 1])

    def test_thompson_numpy(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                 rewards=np.asarray([0, 1, 1, 1, 0, 0, 1, 0, 1]),
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 2, 1, 3])

    def test_thompson_series(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                 rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 3, 3])

    def test_thompson_df(self):

        df = pd.DataFrame({"decisions": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                           "rewards": [0, 0, 1, 0, 1, 0, 1, 1, 1]})

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=df["decisions"],
                                 rewards=df["rewards"],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 1, 3])

    def test_thompson_df_list(self):

        df = pd.DataFrame({"decisions": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                           "rewards": [0, 0, 0, 0, 0, 0, 1, 1, 1]})

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=df["decisions"],
                                 rewards=[0, 0, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 1, 3])

    def test_thompson_non_binary_without_threshold(self):

        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1.25, 1, 1],
                         learning_policy=LearningPolicy.ThompsonSampling(),
                         seed=123456,
                         num_run=4,
                         is_predict=True)

    def test_thompson_binary_with_threshold(self):

        dec_to_threshold = {1: 0.1, 2: 0.1, 3: 0.1}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        self.predict(arms=[1, 2, 3],
                     decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                     rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                     learning_policy=LearningPolicy.ThompsonSampling(binarize),
                     seed=123456,
                     num_run=4,
                     is_predict=True)

    def test_thompson_t1(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 2, 1, 2])

    def test_thompson_t2(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=71,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [1, 3, 3, 1])

    def test_thompson_t3(self):

        dec_to_threshold = {1: 5, 2: 5, 4: 5}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=[1, 2, 4],
                                 decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                 rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [4, 2, 2, 2])

    def test_thompson_t4(self):

        dec_to_threshold = {1: 5, 2: 5, 4: 5}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=[1, 2, 4],
                                 decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                 rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=23,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4, 2, 4])

    def test_thompson_t5(self):

        arms, mab = self.predict(arms=['one', 'two', 'three'],
                                 decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                 rewards=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['three', 'two', 'one', 'three'])

    def test_thompson_t6(self):

        dec_to_threshold = {'one': 3, 'two': 3, 'three': 3}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=['one', 'two', 'three'],
                                 decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                 rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=17,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['one', 'two', 'two', 'two'])

    def test_thompson_t7(self):

        dec_to_threshold = {'a': 1, 'b': 1, 'c': 1}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=['a', 'b', 'c'],
                                 decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                 rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=123456,
                                 num_run=5,
                                 is_predict=True)

        self.assertEqual(arms, ['c', 'b', 'b', 'b', 'b'])

    def test_thompson_t8(self):

        dec_to_threshold = {'a': 1, 'b': 1, 'c': 1}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=['a', 'b', 'c'],
                                 decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                 rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=9,
                                 num_run=5,
                                 is_predict=True)

        self.assertEqual(arms, ['b', 'c', 'c', 'c', 'c'])

    def test_thompson_t9(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        dec_to_threshold = {a: 1, b: 1, c: 1}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=[a, b, c],
                                 decisions=[a, b, c, a, b, c, a, b, c, a],
                                 rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [c, b, c, c])

    def test_thompson_t10(self):

        # Dates to test
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        dec_to_threshold = {a: 1, b: 1, c: 1}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arms, mab = self.predict(arms=[a, b, c],
                                 decisions=[a, b, c, a, b, c, a, b, c, a, b, b],
                                 rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [b, c, b, a])

    def test_fit_twice(self):

        dec_to_threshold = {1: 7, 2: 12, 3: 0}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 2, 1, 1, 2],
                                rewards=[10, 4, 3, 5, 6],
                                learning_policy=
                                LearningPolicy.ThompsonSampling(binarize),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(mab._imp.arm_to_success_count[1], 2)
        self.assertEqual(mab._imp.arm_to_success_count[2], 1)
        self.assertEqual(mab._imp.arm_to_success_count[3], 1)

        self.assertEqual(mab._imp.arm_to_fail_count[1], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[2], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[3], 1)

        self.assertEqual(arm, 3)

        decisions = [1, 2, 3, 1, 2, 3]
        rewards = [1, 0, 3, 7, 11, 22]
        mab.fit(decisions, rewards)

        self.assertEqual(mab._imp.arm_to_success_count[1], 2)
        self.assertEqual(mab._imp.arm_to_success_count[2], 1)
        self.assertEqual(mab._imp.arm_to_success_count[3], 3)

        self.assertEqual(mab._imp.arm_to_fail_count[1], 2)
        self.assertEqual(mab._imp.arm_to_fail_count[2], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[3], 1)

    def test_partial_fit(self):

        dec_to_threshold = {1: 7, 2: 12, 3: 0}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 2, 1, 1, 2],
                                rewards=[10, 4, 3, 5, 6],
                                learning_policy=
                                LearningPolicy.ThompsonSampling(binarize),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(mab._imp.arm_to_success_count[1], 2)
        self.assertEqual(mab._imp.arm_to_success_count[2], 1)
        self.assertEqual(mab._imp.arm_to_success_count[3], 1)

        self.assertEqual(mab._imp.arm_to_fail_count[1], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[2], 3)
        self.assertEqual(mab._imp.arm_to_fail_count[3], 1)

        self.assertEqual(arm, 3)

        decisions = [1, 2, 3, 1, 2, 3]
        rewards = [1, 0, 3, 7, 11, 22]
        mab.partial_fit(decisions, rewards)

        self.assertEqual(mab._imp.arm_to_success_count[1], 3)
        self.assertEqual(mab._imp.arm_to_success_count[2], 1)
        self.assertEqual(mab._imp.arm_to_success_count[3], 3)

        self.assertEqual(mab._imp.arm_to_fail_count[1], 4)
        self.assertEqual(mab._imp.arm_to_fail_count[2], 5)
        self.assertEqual(mab._imp.arm_to_fail_count[3], 1)

    def test_add_arm(self):

        dec_to_threshold = {1: 7, 2: 12, 3: 0}

        def binarize(dec, reward):
            return reward >= dec_to_threshold[dec]

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 2, 1, 1, 2],
                                rewards=[10, 4, 3, 5, 6],
                                learning_policy=
                                LearningPolicy.ThompsonSampling(binarize),
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        mab.add_arm(4)
        self.assertTrue(4 in mab.arms)
        self.assertTrue(4 in mab._imp.arms)
        self.assertTrue(mab._imp.arm_to_fail_count[4] == 1)
        self.assertTrue(mab._imp.arm_to_success_count[4] == 1)

        dec_to_threshold = {1: 7, 2: 12, 3: 0, 4: 7}
        mab.partial_fit([4, 4, 4], [8, 0, 1])

        self.assertTrue(mab._imp.arm_to_fail_count[4] == 3)
        self.assertTrue(mab._imp.arm_to_success_count[4] == 2)

    def test_add_arm_new_function(self):
        def bin1(dec, reward):
            if dec == 0:
                if reward > 50:
                    return 1
                else:
                    return 0
            elif dec == 1:
                if reward < 20:
                    return 1
                else:
                    return 0

        arm, mab = self.predict(arms=[0, 1],
                                decisions=[1, 0, 1, 1, 0],
                                rewards=[10, 4, 3, 70, 6],
                                learning_policy=LearningPolicy.ThompsonSampling(bin1),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertIs(mab._imp.binarizer, bin1)

        def bin2(dec, reward):
            if dec == 0:
                if reward > 50:
                    return 1
                else:
                    return 0
            elif dec == 1:
                if reward < 20:
                    return 1
                else:
                    return 0
            elif dec == 2:
                if reward >= 1:
                    return 1
                else:
                    return 0

        mab.add_arm(2, bin2)

        self.assertTrue(mab._imp.arm_to_fail_count[2] == 1)
        self.assertTrue(mab._imp.arm_to_success_count[2] == 1)
        self.assertIs(mab._imp.binarizer, bin2)

    def test_remove_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.arm_to_expectation)
        self.assertTrue(3 not in mab._imp.arm_to_fail_count)
        self.assertTrue(3 not in mab._imp.arm_to_success_count)

    def test_warm_start(self):

        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[1, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.ThompsonSampling(),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_fail_count, {1: 3, 2: 4, 3: 1})
        self.assertDictEqual(mab._imp.arm_to_success_count, {1: 5, 2: 1, 3: 1})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_fail_count, {1: 3, 2: 4, 3: 3})
        self.assertDictEqual(mab._imp.arm_to_success_count, {1: 5, 2: 1, 3: 5})

    def test_double_warm_start(self):
        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[1, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.ThompsonSampling(),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_fail_count, {1: 3, 2: 4, 3: 1})
        self.assertDictEqual(mab._imp.arm_to_success_count, {1: 5, 2: 1, 3: 1})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_fail_count, {1: 3, 2: 4, 3: 3})
        self.assertDictEqual(mab._imp.arm_to_success_count, {1: 5, 2: 1, 3: 5})

        # Warm start again, #3 is closest to #2 but shouldn't get warm started again
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_fail_count, {1: 3, 2: 4, 3: 3})
        self.assertDictEqual(mab._imp.arm_to_success_count, {1: 5, 2: 1, 3: 5})

    def test_ts_contexts(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 contexts=[[]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [3, 1, 1, 1, 1, 3, 1, 1, 1, 1])

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 contexts=[[1, 2, 3]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [3, 1, 1, 1, 1, 3, 1, 1, 1, 1])
