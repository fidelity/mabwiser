# -*- coding: utf-8 -*-
import math

import numpy as np

from sklearn.preprocessing import StandardScaler

from mabwiser.mab import LearningPolicy
from mabwiser.linear import _RidgeRegression, fix_small_variance
from tests.test_base import BaseTest


class RidgeRegressionTest(BaseTest):

    def test_predict_ridge(self):
        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
        rewards = np.array([3, 3, 1])
        rng = np.random.RandomState(seed=7)

        ridge = _RidgeRegression(rng, l2_lambda=1.0, alpha=1.0, scale=False)

        ridge.init(context.shape[1])
        ridge.fit(context, rewards)
        prediction = ridge.predict(np.array([0, 1, 2, 3, 5]))
        self.assertTrue(math.isclose(prediction, 2.8167701863354, abs_tol=1e-8))

    def test_predict_ridge_scaler(self):
        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
        rewards = np.array([3, 3, 1])
        rng = np.random.RandomState(seed=7)
        scaler = StandardScaler()
        scaler.fit(context.astype('float64'))

        ridge = _RidgeRegression(rng, l2_lambda=1.0, alpha=1.0, scale=True)

        ridge.init(context.shape[1])
        ridge.fit(context, rewards)
        prediction = ridge.predict(np.array([[0, 1, 2, 3, 5]]))
        self.assertTrue(math.isclose(prediction, 1.1429050092142725, abs_tol=1e-8))

    def test_fit(self):

        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
        rewards = np.array([3, 3, 1])
        decisions = np.array([1, 1, 1])

        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09161491, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 0.00310559, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.97515528, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.32142857, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], -0.02018634, abs_tol=0.00000001))

        context2 = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1],
                            [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0], [2, 1, 1, 1, 2],
                            [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1]])
        rewards2 = np.array([3, 3, 1, 0, -1, 2, 1, 2, 1, 1, 0, 3])
        decisions2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions2,
                                 rewards=rewards2,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                 context_history=context2,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09927202, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], -0.17141953, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.09091367, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], -0.03705452, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.59027579, abs_tol=0.00000001))

    def test_fit_twice(self):
        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
        rewards = np.array([3, 3, 1])
        decisions = np.array([1, 1, 1])

        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09161491, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 0.00310559, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.97515528, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.32142857, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], -0.02018634, abs_tol=0.00000001))

        context2 = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1],
                             [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0], [2, 1, 1, 1, 2],
                             [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1],
                             [0, 2, 9, 5, 1]])
        rewards2 = np.array([3, 3, 1, 0, -1, 2, 1, 2, 1, 1, 0, 3, 1])
        decisions2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

        mab.fit(decisions2, rewards2, context2)
        arms = mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09927202, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], -0.17141953, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.09091367, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], -0.03705452, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.59027579, abs_tol=0.00000001))

    def test_partial_fit(self):

        context = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
        rewards = np.array([3, 2, 1])
        decisions = np.array([1, 1, 1])

        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.47619048, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 0.04761905, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], -0.5952381, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.14285714, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.66666667, abs_tol=0.00000001))
        self.assertEqual(mab._imp.arm_to_model[0].beta[0], 0)
        self.assertEqual(mab._imp.arm_to_model[0].beta[1], 0)
        self.assertEqual(mab._imp.arm_to_model[0].beta[2], 0)
        self.assertEqual(mab._imp.arm_to_model[0].beta[3], 0)
        self.assertEqual(mab._imp.arm_to_model[0].beta[4], 0)

        context2 = np.array([[2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards2 = np.array([1, 1, 1])
        decisions2 = np.array([0, 0, 1])

        mab.partial_fit(decisions2, rewards2, context2)

        self.assertEqual(mab._imp.num_features, 5)

        self.assertTrue(math.isclose(mab._imp.arm_to_model[0].beta[0], 0.11940299, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[0].beta[1], 0.01492537, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[0].beta[2], 0.11940299, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[0].beta[3], 0.04477612, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[0].beta[4], 0.17910448, abs_tol=0.00000001))

        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.53019146, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 0.13402062, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], -0.56553756, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.17525773, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.61266568, abs_tol=0.00000001))

    def test_partial_vs_batch_fit(self):

        # Batch fit
        context_batch = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],
                                  [2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards_batch = np.array([0, 1, 1, 0, 1, 0])
        decisions_batch = np.array([1, 1, 1, 0, 0, 1])

        arms_batch, mab_batch = self.predict(arms=[0, 1],
                                             decisions=decisions_batch,
                                             rewards=rewards_batch,
                                             learning_policy=LearningPolicy.LinUCB(alpha=1),
                                             context_history=context_batch,
                                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                             seed=123456,
                                             num_run=1,
                                             is_predict=True)

        # Partial fit
        context = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
        rewards = np.array([0, 1, 1])
        decisions = np.array([1, 1, 1])

        arms_partial, mab_partial = self.predict(arms=[0, 1],
                                                 decisions=decisions,
                                                 rewards=rewards,
                                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                                 context_history=context,
                                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                                 seed=123456,
                                                 num_run=1,
                                                 is_predict=True)

        context2 = np.array([[2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards2 = np.array([0, 1, 0])
        decisions2 = np.array([0, 0, 1])

        mab_partial.partial_fit(decisions2, rewards2, context2)

        self.assertListEqual(mab_batch._imp.arm_to_model[0].beta.tolist(),
                             mab_partial._imp.arm_to_model[0].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].Xty.tolist(), mab_partial._imp.arm_to_model[0].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[0].A_inv.tolist())

        self.assertListEqual(mab_batch._imp.arm_to_model[1].beta.tolist(),
                             mab_partial._imp.arm_to_model[1].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].Xty.tolist(), mab_partial._imp.arm_to_model[1].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[1].A_inv.tolist())

    def test_partial_different_order(self):

        # Batch fit
        context_batch = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],
                                  [2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards_batch = np.array([0, 1, 1, 0, 1, 0])
        decisions_batch = np.array([1, 1, 1, 0, 0, 1])

        arms_batch, mab_batch = self.predict(arms=[0, 1],
                                             decisions=decisions_batch,
                                             rewards=rewards_batch,
                                             learning_policy=LearningPolicy.LinUCB(alpha=1),
                                             context_history=context_batch,
                                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                             seed=123456,
                                             num_run=1,
                                             is_predict=True)

        # Partial fit
        context = np.array([[2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards = np.array([0, 1, 0])
        decisions = np.array([0, 0, 1])

        arms_partial, mab_partial = self.predict(arms=[0, 1],
                                                 decisions=decisions,
                                                 rewards=rewards,
                                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                                 context_history=context,
                                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                                 seed=123456,
                                                 num_run=1,
                                                 is_predict=True)

        context2 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
        rewards2 = np.array([0, 1, 1])
        decisions2 = np.array([1, 1, 1])

        mab_partial.partial_fit(decisions2, rewards2, context2)
        self.assertListEqual(mab_batch._imp.arm_to_model[0].beta.tolist(),
                             mab_partial._imp.arm_to_model[0].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].Xty.tolist(), mab_partial._imp.arm_to_model[0].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[0].A_inv.tolist())

        self.assertListEqual(mab_batch._imp.arm_to_model[1].beta.tolist(),
                             mab_partial._imp.arm_to_model[1].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].Xty.tolist(), mab_partial._imp.arm_to_model[1].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[1].A_inv.tolist())

    def test_batch_vs_3_partial_fit(self):

        # Batch fit
        context_batch = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2],
                                  [2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1],
                                  [2, 2, 2, 2, 1], [1, 2, 3, 1, 1]])
        rewards_batch = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        decisions_batch = np.array([1, 1, 1, 0, 0, 1, 0, 0])

        arms_batch, mab_batch = self.predict(arms=[0, 1],
                                             decisions=decisions_batch,
                                             rewards=rewards_batch,
                                             learning_policy=LearningPolicy.LinUCB(alpha=1),
                                             context_history=context_batch,
                                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                             seed=123456,
                                             num_run=1,
                                             is_predict=True)

        # Partial fit
        context = np.array([[2, 2, 2, 2, 1], [1, 2, 3, 1, 1]])
        rewards = np.array([1, 1])
        decisions = np.array([0, 0])

        arms_partial, mab_partial = self.predict(arms=[0, 1],
                                                 decisions=decisions,
                                                 rewards=rewards,
                                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                                 context_history=context,
                                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                                 seed=123456,
                                                 num_run=1,
                                                 is_predict=True)

        context2 = np.array([[1, 0, 0, 0, 1], [0, 1, 2, 3, 4], [2, 0, 1, 0, 2]])
        rewards2 = np.array([0, 1, 1])
        decisions2 = np.array([1, 1, 1])

        context3 = np.array([[2, 1, 2, 1, 2], [3, 3, 3, 2, 1], [1, 1, 1, 1, 1]])
        rewards3 = np.array([0, 1, 0])
        decisions3 = np.array([0, 0, 1])

        mab_partial.partial_fit(decisions2, rewards2, context2)
        mab_partial.partial_fit(decisions3, rewards3, context3)

        self.assertListEqual(mab_batch._imp.arm_to_model[0].beta.tolist(),
                             mab_partial._imp.arm_to_model[0].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].Xty.tolist(), mab_partial._imp.arm_to_model[0].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[0].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[0].A_inv.tolist())

        self.assertListEqual(mab_batch._imp.arm_to_model[1].beta.tolist(),
                             mab_partial._imp.arm_to_model[1].beta.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].Xty.tolist(), mab_partial._imp.arm_to_model[1].Xty.tolist())
        self.assertListEqual(mab_batch._imp.arm_to_model[1].A_inv.tolist(),
                             mab_partial._imp.arm_to_model[1].A_inv.tolist())

    def test_l2_low(self):

        context = np.array([[1, 1, 0, 0, 1], [0, 1, 2, 9, 4], [2, 3, 1, 0, 2]])
        rewards = np.array([3, 2, 1])
        decisions = np.array([1, 1, 1])

        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=0.1),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [1, 1])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 1.59499705, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], -0.91856183, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], -2.49775977, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.14219195, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 1.65819347, abs_tol=0.00000001))

    def test_l2_high(self):

        context = np.array([[1, 1, 0, 0, 1], [0, 1, 2, 9, 4], [2, 3, 1, 0, 2]])
        rewards = np.array([3, 2, 1])
        decisions = np.array([1, 1, 1])
        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=10),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.18310155, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 0.16372811, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], -0.00889076, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], 0.09434416, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.22503229, abs_tol=0.00000001))

    def test_l2_0(self):

        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1],
                            [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0], [2, 1, 1, 1, 2],
                            [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1]])
        rewards = np.array([3, 3, 1, 0, -1, 2, 1, 2, 1, 1, 0, 3])
        decisions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=0),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [1, 1])

        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09224215, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], -0.20569848, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.13434242, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], -0.1000045, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.63726682, abs_tol=0.00000001))

        context2 = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1], [1, 2, 3, 4, 5]])
        rewards2 = np.array([-1, 2, 1, 2, 0])
        decisions2 = np.array([1, 1, 1, 1, 1])
        mab.fit(decisions2, rewards2, context2)

        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.97297297, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], 1.05405405, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], -0.86486486, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], -0.72972973, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.48648649, abs_tol=0.00000001))

    def test_fit_twice_new_features(self):

        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2], [-1, 4, 2, 0, 1],
                            [2, 2, 2, 2, 2], [3, 2, 1, 2, 3], [0, 0, 0, 0, 0], [2, 1, 1, 1, 2],
                            [3, 2, 3, 2, 3], [8, 2, 3, 1, 0], [1, 2, -9, -7, 1], [0, 1, 1, 1, 1]])
        rewards = np.array([3, 3, 1, 0, -1, 2, 1, 2, 1, 1, 0, 3])
        decisions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=0),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [1, 1])

        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[0], 0.09224215, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[1], -0.20569848, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[2], 0.13434242, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[3], -0.1000045, abs_tol=0.00000001))
        self.assertTrue(math.isclose(mab._imp.arm_to_model[1].beta[4], 0.63726682, abs_tol=0.00000001))

        context2 = np.array([[1, 0, 2, 1, 1, 3], [3, 1, 2, 3, 4, 1], [2, -1, 1, 0, 2, 2], [-1, 4, 2, 0, 1, 0],
                             [1, 2, 3, 4, 5, 1]])
        rewards2 = np.array([-1, 2, 1, 2, 0])
        decisions2 = np.array([1, 1, 1, 1, 1])
        mab.fit(decisions2, rewards2, context2)
        self.assertEqual(mab._imp.num_features, 6)

    def test_add_arm(self):

        context = np.array([[1, 0, 2, 1, 1], [3, 1, 2, 3, 4], [2, -1, 1, 0, 2]])
        rewards = np.array([3, 3, 1])
        decisions = np.array([1, 1, 1])
        arms, mab = self.predict(arms=[0, 1],
                                 decisions=decisions,
                                 rewards=rewards,
                                 learning_policy=LearningPolicy.LinUCB(alpha=1),
                                 context_history=context,
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(mab._imp.num_features, 5)
        self.assertEqual(arms, [0, 0])
        mab.add_arm(2)
        self.assertTrue(2 in mab._imp.arm_to_model.keys())
        self.assertEqual(mab._imp.arm_to_model[2].beta[0], 0)
        self.assertEqual(mab._imp.arm_to_model[2].beta[1], 0)
        self.assertEqual(mab._imp.arm_to_model[2].beta[2], 0)
        self.assertEqual(mab._imp.arm_to_model[2].beta[3], 0)
        self.assertEqual(mab._imp.arm_to_model[2].beta[4], 0)

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
        self.assertListAlmostEqual(mab._imp.arm_to_model[1].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta, [0, 0, 0, 0, 0])

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0, 1]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])

        # Warm start again, #3 shouldn't change even though it's closer to #2 now
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0.5, 0.5], 3: [0.5, 0.5]}, distance_quantile=0.5)
        self.assertListAlmostEqual(mab._imp.arm_to_model[3].beta,
                                   [0.19635284, 0.11556404, 0.57675997, 0.30597964, -0.39100933])

    def test_fix_small_variance(self):
        rng = np.random.default_rng(1234)
        context = rng.random((10000, 10))

        # Set first feature to have variance close to zero
        context[0, 0] = 0.0001
        context[1:, 0] = [0] * (10000 - 1)

        scaler = StandardScaler()
        scaler.fit(context)

        self.assertAlmostEqual(scaler.scale_[0], 9.99949999e-07)
        self.assertAlmostEqual(scaler.var_[0], 9.99900000e-13)

        # Fix small variance
        fix_small_variance(scaler)

        self.assertAlmostEqual(scaler.scale_[0], 1)
        self.assertAlmostEqual(scaler.var_[0], 0)
