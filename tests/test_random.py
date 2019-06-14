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

        self.assertEqual(arm, 1)

    def test_random_expectations(self):
        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=7,
                                num_run=1,
                                is_predict=False)
        self.assertDictEqual(exp, {1: 0, 2: 0, 3: 0})

    def test_random_seed(self):
        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

        arm, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                learning_policy=LearningPolicy.Random(),
                                seed=23,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 3)

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

        self.assertEqual(arm, 1)
