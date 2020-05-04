# -*- coding: utf-8 -*-

import numpy as np
from mabwiser.mab import LearningPolicy, NeighborhoodPolicy, _Popularity
from tests.test_base import BaseTest


class PopularityTest(BaseTest):

    def test(self):
        import mabwiser.mab as mab
        print(mab.__version__)

    def test_2arm_equal_prob(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 2, 1, 2, 1])

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        # Check that normalized probabilities are 1/2 each, and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2])
        self.assertAlmostEqual(exp[1], exp[2])

    def test_2arm_diff_prob(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 0, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        # print(arm)
        self.assertEqual(arm, [1, 2, 1, 2, 1])

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 0, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        # print(exp)
        # Check that normalized probabilities are 0.66 and 0.33, and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2])
        self.assertAlmostEqual(exp[1], exp[2]*2)

    def test_2arm_diff_prob_2(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        # print(arm)
        self.assertEqual(arm, [1, 2, 1, 2, 1])

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[1, 1, 1, 0, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        # print(exp)
        # Check that normalized probabilities are 0.75 and 0.25, and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2])
        self.assertAlmostEqual(exp[1], exp[2] * 3)

    def test_3arm_equal_prob(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)

        self.assertEqual(arm, [1, 3, 1, 3, 2])

        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        # Check that normalized probabilities are 1/3 each, and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2] + exp[3])
        self.assertAlmostEqual(exp[1], exp[2])
        self.assertAlmostEqual(exp[2], exp[3])
        self.assertAlmostEqual(exp[3], exp[1])

    def test_3arm_diff_prob(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 0, 0, 1, 1, 1, 1, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        # print(arm)
        self.assertEqual(arm, [1, 3, 2, 3, 2])

        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[1, 0, 0, 1, 1, 1, 1, 0, 1],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)

        # print(exp)
        # Check that normalized probabilities are 0.16, 0.5, 0.33 and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2] + exp[3])
        self.assertAlmostEqual(exp[2], 0.5)
        self.assertAlmostEqual(exp[3], exp[1]*2)

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
        # print("arm:", arm)
        # Result for each run. At each we have 2 context to predict
        self.assertListEqual(arm[0], [3, 1])
        self.assertListEqual(arm[1], [1, 1])
        self.assertListEqual(arm[2], [1, 1])

        exp, mab = self.predict(arms=[1, 2, 3, 4],
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
                                num_run=1,
                                is_predict=False)
        # print("exp:", exp)
        # Result for each context should have expectations/probs that sum up to 1.0
        self.assertAlmostEqual(1.0, exp[0][1] + exp[0][2] + exp[0][3] + exp[0][4])
        self.assertAlmostEqual(1.0, exp[1][1] + exp[1][2] + exp[1][3] + exp[1][4])

        # {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}]
        self.assertAlmostEqual(exp[0][1], exp[0][2])
        self.assertAlmostEqual(exp[0][2], exp[0][3])
        self.assertAlmostEqual(exp[0][3], exp[0][4])

        # {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}]
        self.assertAlmostEqual(exp[1][1], 1.0)

    def test_zero_reward(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[0, 0, 0, 0, 0, 0],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=5,
                                is_predict=True)
        # print(arm)
        self.assertEqual(arm, [1, 2, 1, 2, 1])

        exp, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[0, 0, 0, 0, 0, 0],
                                learning_policy=LearningPolicy.Popularity(),
                                seed=123456,
                                num_run=1,
                                is_predict=False)
        # print(exp)
        # Check that normalized probabilities are equal and sum up to 1.0
        self.assertAlmostEqual(1.0, exp[1] + exp[2])
        self.assertAlmostEqual(exp[1], exp[2])

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
        self.assertDictEqual({'Arm1': 0.38016528925619836, 'Arm2': 0.6198347107438016},
                             mab.predict_expectations())

        # Hack into epsilon from underlying greedy bandit
        mab.epsilon = 5
        mab.fit(np.array(decisions), np.array(rewards))

        # Assert epsilon change has no impact
        # self.assertEqual("Arm1", mab.predict())
        self.assertDictEqual({'Arm1': 0.38016528925619836, 'Arm2': 0.6198347107438016},
                             mab.predict_expectations())




