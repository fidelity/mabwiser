# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class GreedyTest(BaseTest):

    def test_epsilon_0(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                                seed=7,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 3)

    def test_epsilon_0_missing_decision(self):

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 3, 3, 3],
                                rewards=[0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                seed=7,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 3)

    def test_epsilon_0_multiple_prediction(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 seed=7,
                                 num_run=5,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 3, 3, 3, 3])

    def test_epsilon_50(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=7,
                                 num_run=5,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 3, 3, 2, 3])

    def test_seed_epsilon50(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=123456,
                                 num_run=5,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 2, 3, 3, 3])

        # change seed and assert a different result
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=123,
                                 num_run=5,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1, 3, 3, 2])

    def test_predict_expectation(self):

        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[10, 20, 30, -10, 0, 16, 2, 7, 3],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertDictEqual(exps, {1: 20.0, 2: 2.0, 3: 4.0})

    def test_epsilon25_numpy(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                 rewards=np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 3, 2])

    def test_epsilon25_series(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                 rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 3, 2])

    def test_epsilon25_df(self):

        df = pd.DataFrame({"decisions": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                           "rewards": [0, 0, 0, 0, 0, 0, 1, 1, 1]})

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=df["decisions"],
                                 rewards=df["rewards"],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 3, 2])

    def test_epsilon25_df_list(self):

        df = pd.DataFrame({"decisions": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                           "rewards": [0, 0, 0, 0, 0, 0, 1, 1, 1]})

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=df["decisions"],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [3, 3, 3, 2])

    def test_unused_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=23,
                                 num_run=5,
                                 is_predict=True)

        # not used arm (4) can still be picked up thanks to randomness
        self.assertEqual(arms, [3, 3, 3, 1, 4])

    def test_fit_twice(self):

        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[10, 20, 30, -10, 0, 16, 2, 7, 3],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)
        # First fit
        self.assertDictEqual(exps, {1: 20.0, 2: 2.0, 3: 4.0})

        # Second fit
        mab.fit([1, 1, 2, 2, 3, 3], [2, 4, 4, 6, 6, 8])
        self.assertDictEqual(mab.predict_expectations(), {1: 0.0474454214269846, 2: 0.9552527396157818,
                                                          3: 0.906050936603124})

    def test_mismatch_context(self):

        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=[10, 20, 30, -10, 0, 16, 2, 7, 3],
                         context_history=[[10], [10], [10], [10], [10], [10], [10], [10], [10]],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                         neighborhood_policy=NeighborhoodPolicy.Radius(),
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_greedy_t1(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [1, 1, 1, 1])

    def test_greedy_t2(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=71,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [2, 2, 1, 1])

    def test_greedy_t3(self):

        arms, mab = self.predict(arms=[1, 2, 4],
                                 decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                 rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4, 1, 4])

    def test_greedy_t4(self):

        arms, mab = self.predict(arms=[1, 2, 4],
                                 decisions=[1, 1, 4, 4, 2, 2, 1, 1, 4, 2, 1, 4, 1, 2, 4],
                                 rewards=[7, 9, 10, 20, 2, 5, 8, 15, 17, 11, 0, 5, 2, 9, 3],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=23,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4, 4, 2])

    def test_greedy_t5(self):

        arms, mab = self.predict(arms=['one', 'two', 'three'],
                                 decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                 rewards=[1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['one', 'one', 'one', 'one'])

    def test_greedy_t6(self):

        arms, mab = self.predict(arms=['one', 'two', 'three'],
                                 decisions=['one', 'one', 'one', 'three', 'two', 'two', 'three', 'one', 'three', 'two'],
                                 rewards=[2, 7, 7, 9, 1, 3, 1, 2, 6, 4],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=17,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['three', 'one', 'three', 'two'])

    def test_greedy_t7(self):

        arms, mab = self.predict(arms=['a', 'b', 'c'],
                                 decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                 rewards=[-1.25, 12, 0.7, 10, 12, 9.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['b', 'b', 'a', 'b'])

    def test_greedy_t8(self):

        arms, mab = self.predict(arms=['a', 'b', 'c'],
                                 decisions=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'],
                                 rewards=[-1.25, 0.7, 12, 10, 12, 9.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=9,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, ['c', 'b', 'c', 'c'])

    def test_greedy_t9(self):

        # Dates for testing
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arms, mab = self.predict(arms=[a, b, c],
                                 decisions=[a, b, c, a, b, c, a, b, c, a],
                                 rewards=[1.25, 0.7, 12, 10, 1.43, 0.2, -1, -10, 4, 0],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(arms, [c, c, a, c])

    def test_greedy_t10(self):

        # Dates for testing
        a = datetime.datetime(2018, 1, 1)
        b = datetime.datetime(2017, 7, 31)
        c = datetime.datetime(2018, 9, 15)

        arms, mab = self.predict(arms=[a, b, c],
                                 decisions=[a, b, c, a, b, c, a, b, c, a, b, b],
                                 rewards=[7, 12, 1, -10, 5, 1, 2, 9, 3, 3, 6, 7],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)
        self.assertEqual(arms, [b, b, b, b])

    def test_partial_fit(self):

        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[10, 20, 30, -10, 0, 16, 2, 7, 3],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)
        # First fit
        self.assertDictEqual(exps, {1: 20.0, 2: 2.0, 3: 4.0})

        # Second fit
        mab.partial_fit([1, 1, 2, 2, 3, 3], [2, 4, 4, 6, 6, 8])
        self.assertDictEqual(mab.predict_expectations(), {1: 0.0474454214269846, 2: 0.9552527396157818,
                                                          3: 0.906050936603124})

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.add_arm(4)
        self.assertTrue(4 in mab.arms)
        self.assertTrue(4 in mab._imp.arms)
        self.assertTrue(4 in mab._imp.arm_to_expectation.keys())
        self.assertTrue(mab._imp.arm_to_sum[4] == 0)

    def test_remove_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.arm_to_expectation)

    def test_warm_start(self):

        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.5, 2: 0.0, 3: 0.0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.5, 2: 0.0, 3: 0.5})

    def test_double_warm_start(self):
        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.5, 2: 0.0, 3: 0.0})

        # Warm start, #3 gets warm started by #2
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.5, 2: 0.0, 3: 0.0})

        # Warm start again, #3 is closest to #1 but shouldn't get warm started again
        mab.warm_start(arm_to_features={1: [0, 1], 2: [-1, -1], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 0.5, 2: 0.0, 3: 0.0})

    def test_greedy_contexts(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 contexts=[[]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [1, 1, 1, 1, 1, 1, 3, 1, 1, 1])

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                 contexts=[[1, 2, 3]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [1, 1, 1, 1, 1, 1, 3, 1, 1, 1])
