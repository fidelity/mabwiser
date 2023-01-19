# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy, _Popularity
from tests.test_base import BaseTest


class PopularityTest(BaseTest):

    def test_2arm_equal_prob(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 1, 1, 2, 1])

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        self.assertAlmostEqual(exp[1], 0.9948500327379373)
        self.assertAlmostEqual(exp[2], 0.005149967262062828)

    def test_2arm_diff_prob(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 0, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        self.assertEqual(arm, [1, 1, 1, 2, 1])

    def test_2arm_diff_prob_2(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        self.assertEqual(arm, [1, 1, 1, 2, 1])

    def test_3arm_equal_prob(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        self.assertEqual(arm, [3, 2, 3, 3, 3])

    def test_3arm_diff_prob(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 0, 0, 1, 1, 1, 1, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        self.assertEqual(arm, [3, 2, 3, 3, 3])

    def test_with_context(self):
        arm, mab = self.predict(arms=[1, 2, 3, 4],
                                decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                neighborhood_policy=NeighborhoodPolicy.KNearest(),
                                context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                 [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                 [0, 2, 1, 0, 0]],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                seed=123456,
                                num_run=3,
                                is_predict=True)
        self.assertListEqual(arm[0], [1, 1])
        self.assertListEqual(arm[1], [1, 1])
        self.assertListEqual(arm[2], [3, 1])

    def test_zero_reward(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[0, 0, 0, 0, 0, 0],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        self.assertEqual(arm, [1, 1, 1, 2, 1])

    def test_epsilon_has_no_impact(self):

        # This is super hack test to check that epsilon has no impact
        # on popularity results
        arms = ['Arm1', 'Arm2']
        mab = _Popularity(rng=np.random.RandomState(seed=123456),
                          arms=arms, n_jobs=1, backend=None)
        decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        rewards = [20, 17, 25, 9]
        mab.fit(np.array(decisions), np.array(rewards))

        # Original result
        self.assertDictEqual({'Arm1': 0.03702697841958926, 'Arm2': 0.9629730215804108},
                             mab.predict_expectations())

        # Hack into epsilon from underlying greedy bandit
        mab = _Popularity(rng=np.random.RandomState(seed=123456),
                          arms=arms, n_jobs=1, backend=None)
        mab.epsilon = 5
        mab.fit(np.array(decisions), np.array(rewards))

        # Assert epsilon change has no impact
        # self.assertEqual("Arm1", mab.predict())
        self.assertDictEqual({'Arm1': 0.03702697841958926, 'Arm2': 0.9629730215804108},
                             mab.predict_expectations())

    def test_2arm_partial_fit(self):
        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        # Initial probabilities
        self.assertAlmostEqual(exp[1], 0.9991137956839969)
        self.assertAlmostEqual(exp[2], 0.0008862043160030626)

        # Partial fit
        mab.partial_fit([1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1])
        exp = mab.predict_expectations()

        self.assertAlmostEqual(exp[1], 0.9162612769403672)
        self.assertAlmostEqual(exp[2], 0.08373872305963273)

    def test_fit_twice(self):

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        # Initial probabilities
        self.assertAlmostEqual(exp[1], 0.9991137956839969)
        self.assertAlmostEqual(exp[2], 0.0008862043160030626)

        # Fit the other way around
        mab.fit([2, 2, 2, 1, 1, 1], [1, 1, 1, 0, 1, 1])
        exp = mab.predict_expectations()
        self.assertAlmostEqual(exp[1], 0.9262956187781518)
        self.assertAlmostEqual(exp[2], 0.07370438122184816)

    def test_unused_arm(self):

        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        # Initial probabilities
        self.assertAlmostEqual(exp[3], 0.0)

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Popularity(),
                                 neighborhood_policy=NeighborhoodPolicy.Clusters(2),
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
        self.assertTrue(5 in mab._imp.lp_list[0].arms)
        self.assertTrue(5 in mab._imp.lp_list[0].arm_to_expectation.keys())

    def test_string_arms(self):

        arms, mab = self.predict(arms=["one", "two"],
                                 decisions=["one", "one", "one", "two", "two", "two"],
                                 rewards=[1, 1, 1, 0, 1, 1],
                                 learning_policy=LearningPolicy.Popularity(),
                                 seed=123456,
                                 num_run=5,
                                 is_predict=True)
        self.assertEqual(arms, ['one', 'one', 'one', 'two', 'one'])

    def test_different_seeds(self):

        arm, mab = self.predict(arms=["one", "two"],
                                decisions=["one", "one", "one", "two", "two", "two"],
                                rewards=[1, 1, 1, 0, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)
        self.assertEqual('one', arm)

        # Same setup but change seed that prefers the other arm
        arm, mab = self.predict(arms=["one", "two"],
                                decisions=["one", "one", "one", "two", "two", "two"],
                                rewards=[1, 1, 1, 0, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=12234,
                                num_run=1,
                                is_predict=True)
        self.assertEqual('two', arm)

    def test_numpy_rewards(self):

        exp, mab = self.predict(arms=["one", "two"],
                                decisions=["one", "one", "one", "two", "two", "two"],
                                rewards=np.array([1, 1, 1, 0, 1, 1]),
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        self.assertAlmostEqual(exp["one"], 0.9991137956839969)
        self.assertAlmostEqual(exp["two"], 0.0008862043160030626)

    def test_data_frame(self):

        df = pd.DataFrame({"decisions": ["one", "one", "one", "two", "two", "two"],
                           "rewards": [1, 1, 1, 0, 1, 1]})

        exp, mab = self.predict(arms=["one", "two"],
                                decisions=df["decisions"],
                                rewards=df["rewards"],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        self.assertAlmostEqual(exp["one"], 0.9991137956839969)
        self.assertAlmostEqual(exp["two"], 0.0008862043160030626)

    def test_negative_rewards(self):
        with self.assertRaises(ValueError):
            arm, mab = self.predict(arms=[1, 2],
                                    decisions=[1, 1, 1, 2, 2, 2],
                                    rewards=[-1, -1, 1, 1, 1, 1],
                                    learning_policy=LearningPolicy.Popularity(),
                                    seed=123456,
                                    num_run=5,
                                    is_predict=True)
            self.assertEqual(arm, [2, 2, 2, 2, 2])

    def test_remove_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Popularity(),
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
                              learning_policy=LearningPolicy.Popularity(),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 1.0, 2: 0.0, 3: 0.0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 1.0, 2: 0.0, 3: 1.0})

    def test_double_warm_start(self):

        _, mab = self.predict(arms=[1, 2, 3],
                              decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                              rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                              learning_policy=LearningPolicy.Popularity(),
                              seed=7,
                              num_run=1,
                              is_predict=False)

        # Before warm start
        self.assertEqual(mab._imp.trained_arms, [1, 2])
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 1.0, 2: 0.0, 3: 0.0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 1.0, 2: 0.0, 3: 1.0})

        # Warm start again, #3 shouldn't change even though it's closer to #2 now
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertDictEqual(mab._imp.arm_to_expectation, {1: 1.0, 2: 0.0, 3: 1.0})

    def test_popularity_contexts(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Popularity(),
                                 contexts=[[]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [3, 2, 3, 3, 3, 2, 2, 3, 2, 3])

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Popularity(),
                                 contexts=[[1, 2, 3]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [3, 2, 3, 3, 3, 2, 2, 3, 2, 3])
