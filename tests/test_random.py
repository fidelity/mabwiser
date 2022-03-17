# -*- coding: utf-8 -*-

from mabwiser.mab import LearningPolicy
from tests.test_base import BaseTest


class RandomTest(BaseTest):

    def test_random(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=7,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

    def test_random_expectations(self):
        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=7,
                                num_run=1,
                                is_predict=False)
        self.assertDictEqual(exp, {1: 0.625095466604667, 2: 0.8972138009695755, 3: 0.7756856902451935})

    def test_random_seed(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=23,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=72,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=654321,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 3)

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Random(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.add_arm(4)
        self.assertTrue(4 in mab.arms)
        self.assertTrue(4 in mab._imp.arm_to_expectation)

    def test_remove_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Random(),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arm_to_expectation)

    def test_random_contexts(self):
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Random(),
                                 contexts=[[]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [1, 1, 3, 1, 2, 2, 2, 2, 2, 2])

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                                 rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Random(),
                                 contexts=[[1, 2, 3]] * 10,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertEqual(arms, [1, 1, 3, 1, 2, 2, 2, 2, 2, 2])
