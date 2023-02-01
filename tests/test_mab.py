# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import pandas as pd

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class MABTest(BaseTest):

    #################################################
    # Test property decorator methods
    ################################################

    def test_learning_policy_property(self):
        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)
            test_lp = mab.learning_policy
            self.assertTrue(type(test_lp) is type(lp))

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2], para_lp)
            test_lp = mab.learning_policy
            self.assertTrue(type(test_lp) is type(para_lp))

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:
                mab = MAB([1, 2], lp, cp)
                test_lp = mab.learning_policy
                self.assertTrue(type(test_lp) is type(lp))

        for cp in BaseTest.cps:
            for para_lp in BaseTest.lps:
                mab = MAB([1, 2], para_lp, cp)
                test_lp = mab.learning_policy
                self.assertTrue(type(test_lp) is type(para_lp))

    def test_learning_policy_values(self):
        lp = LearningPolicy.EpsilonGreedy(epsilon=0.6)
        mab = MAB([0, 1], lp)
        self.assertEqual(lp.epsilon, mab.learning_policy.epsilon)

        lp = LearningPolicy.LinUCB(alpha=2.0, l2_lambda=0.3, scale=True)
        mab = MAB([0, 1], lp)
        self.assertEqual(lp.alpha, mab.learning_policy.alpha)
        self.assertEqual(lp.l2_lambda, mab.learning_policy.l2_lambda)
        self.assertEqual(lp.scale, mab.learning_policy.scale)

        lp = LearningPolicy.Softmax(tau=0.5)
        mab = MAB([0, 1], lp)
        self.assertEqual(lp.tau, mab.learning_policy.tau)

        def binary(arm, reward):
            return reward == 1

        lp = LearningPolicy.ThompsonSampling(binarizer=binary)
        mab = MAB([0, 1], lp)
        self.assertIs(lp.binarizer, mab.learning_policy.binarizer)

        lp = LearningPolicy.UCB1(alpha=0.7)
        mab = MAB([0, 1], lp)
        self.assertEqual(lp.alpha, mab.learning_policy.alpha)

    def test_neighborhood_policy_property(self):
        for cp in BaseTest.cps:
            for lp in BaseTest.lps:
                mab = MAB([1, 2], lp, cp)
                test_np = mab.neighborhood_policy
                self.assertTrue(type(test_np) is type(cp))

        for cp in BaseTest.cps:
            for para_lp in BaseTest.lps:
                mab = MAB([1, 2], para_lp, cp)
                test_np = mab.neighborhood_policy
                self.assertTrue(type(test_np) is type(cp))

    def test_neighborhood_policy_values(self):
        lp = LearningPolicy.EpsilonGreedy()
        np = NeighborhoodPolicy.Clusters(n_clusters=3)
        mab = MAB([0, 1], lp, np)
        self.assertEqual(np.n_clusters, mab.neighborhood_policy.n_clusters)
        self.assertFalse(mab.neighborhood_policy.is_minibatch)

        np = NeighborhoodPolicy.Clusters(n_clusters=5, is_minibatch=True)
        mab = MAB([0, 1], lp, np)
        self.assertEqual(np.n_clusters, mab.neighborhood_policy.n_clusters)
        self.assertTrue(mab.neighborhood_policy.is_minibatch)

        np = NeighborhoodPolicy.KNearest(k=10, metric='cityblock')
        mab = MAB([0, 1], lp, np)
        self.assertEqual(np.k, mab.neighborhood_policy.k)
        self.assertEqual(np.metric, mab.neighborhood_policy.metric)

        np = NeighborhoodPolicy.Radius(radius=1.5, metric='canberra', no_nhood_prob_of_arm=[0.2, 0.8])
        mab = MAB([0, 1], lp, np)
        self.assertEqual(np.radius, mab.neighborhood_policy.radius)
        self.assertEqual(np.metric, mab.neighborhood_policy.metric)
        self.assertEqual(np.no_nhood_prob_of_arm, mab.neighborhood_policy.no_nhood_prob_of_arm)

        np = NeighborhoodPolicy.LSHNearest(n_dimensions=2, n_tables=2, no_nhood_prob_of_arm=[0.2, 0.8])
        mab = MAB([0, 1], lp, np)
        self.assertEqual(np.n_dimensions, mab.neighborhood_policy.n_dimensions)
        self.assertEqual(np.n_tables, mab.neighborhood_policy.n_tables)
        self.assertEqual(np.no_nhood_prob_of_arm, mab.neighborhood_policy.no_nhood_prob_of_arm)

    #################################################
    # Test context free predict() method
    ################################################

    def test_arm_list_int(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_arm_list_str(self):

        for lp in MABTest.lps:
            self.predict(arms=["A", "B", "C"],
                         decisions=["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=["A", "B", "C"],
                         decisions=["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_series_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_decision_array_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    #################################################
    # Test context free predict_expectation() method
    ################################################

    def test_exp_arm_list_int(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_arm_list_str(self):

        for lp in MABTest.lps:
            self.predict(arms=["A", "B", "C"],
                         decisions=["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_series_reward_array(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_exp_decision_array_reward_series(self):

        for lp in MABTest.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                         rewards=pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         seed=123456,
                         num_run=1,
                         is_predict=False)

    def test_context_history_series(self):

        contexts = pd.DataFrame({'column1': [1, 2, 3], 'column2': [2, 3, 1]})

        for lp in BaseTest.para_lps:
            arm, mab = self.predict(arms=[0, 1],
                                    decisions=[1, 1, 1],
                                    rewards=[0, 0, 0],
                                    learning_policy=lp,
                                    context_history=contexts['column1'],
                                    contexts=[[1]],
                                    seed=123456,
                                    num_run=1,
                                    is_predict=True)

            self.assertEqual(mab._imp.arm_to_model[0].beta.shape[0], 1)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp) or isinstance(cp, NeighborhoodPolicy.TreeBandit):
                    continue

                arm, mab = self.predict(arms=[0, 1],
                                        decisions=[1, 1, 1],
                                        rewards=[0, 0, 0],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=contexts['column1'],
                                        contexts=[[1]],
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)

                # Tree Bandit does not store contexts
                if not isinstance(cp, NeighborhoodPolicy.TreeBandit):
                    self.assertEqual(np.ndim(mab._imp.contexts), 2)

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                arm, mab = self.predict(arms=[0, 1],
                                        decisions=[1, 1, 1],
                                        rewards=[0, 0, 0],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=contexts['column1'],
                                        contexts=[[1]],
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)

                # Tree Bandit does not store contexts
                if not isinstance(cp, NeighborhoodPolicy.TreeBandit):
                    self.assertEqual(np.ndim(mab._imp.contexts), 2)

    def test_context_series(self):

        contexts = pd.DataFrame({'column1': [1, 2, 3, 3, 2, 1], 'column2': [2, 3, 1, 1, 2, 3]})

        for lp in BaseTest.para_lps:
            arm, mab = self.predict(arms=[0, 1],
                                    decisions=[1, 1, 1, 1, 1, 1],
                                    rewards=[0, 0, 0, 0, 0, 0],
                                    learning_policy=lp,
                                    context_history=contexts['column1'],
                                    contexts=pd.Series([1]),
                                    seed=123456,
                                    num_run=1,
                                    is_predict=True)

            self.assertEqual(mab._imp.arm_to_model[0].beta.shape[0], 1)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                arm, mab = self.predict(arms=[0, 1],
                                        decisions=[1, 1, 1, 1, 1, 1],
                                        rewards=[0, 0, 0, 0, 0, 0],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=contexts['column1'],
                                        contexts=pd.Series([1]),
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)

                # Tree Bandit does not store contexts
                if not isinstance(cp, NeighborhoodPolicy.TreeBandit):
                    self.assertEqual(np.ndim(mab._imp.contexts), 2)

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                arm, mab = self.predict(arms=[0, 1],
                                        decisions=[1, 1, 1, 1, 1, 1],
                                        rewards=[0, 0, 0, 0, 0, 0],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=contexts['column1'],
                                        contexts=pd.Series([1]),
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)

                # Tree Bandit does not store contexts
                if not isinstance(cp, NeighborhoodPolicy.TreeBandit):
                    self.assertEqual(np.ndim(mab._imp.contexts), 2)

    #################################################
    # Test contextual predict() method
    ################################################

    def test_context_arm_list_int(self):

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:
                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_arm_list_str(self):

        for lp in MABTest.para_lps:
            self.predict(arms=["A", "B", "C", "D"],
                         decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=["A", "B", "C", "D"],
                             decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=["A", "B", "C", "D"],
                             decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, -2, 2, 3, 11], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, -5, 2, 3, 10], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, -2, 4, 3, 9], [20, 19, 18, 17, 16], [1, 2, 1, 1, 3],
                                              [17, 18, 17, 19, 18]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_decision_series(self):

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_reward_series(self):
        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_decision_reward_series(self):
        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                         rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=pd.Series([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_decision_array(self):

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_reward_array(self):

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    def test_context_decision_reward_array(self):

        for lp in MABTest.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                         rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                         learning_policy=lp,
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                          [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                          [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                          [0, 2, 1, 0, 0]],
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                self.predict(arms=[1, 2, 3, 4],
                             decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                             rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                              [0, 2, 1, 0, 0]],
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             seed=123456,
                             num_run=1,
                             is_predict=True)

    #################################################
    # Test random generator
    ################################################
    def test_seed(self):

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 1, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 3, 2])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=79,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 1, 1, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 1, 2])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 3, 2])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=79,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 1, 1, 3])
        self.assertIs(mab._rng, mab._imp.rng)

    def test_set_rng(self):
        for lp in MABTest.lps:
            mab = MAB([0, 1], lp)
            self.assertIs(mab._rng, mab._imp.rng)

        for lp in MABTest.para_lps:
            mab = MAB([0, 1], lp)
            self.assertIs(mab._rng, mab._imp.rng)

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([0, 1], lp, cp)
                self.assertIs(mab._rng, mab._imp.rng)
                self.assertIs(mab._rng, mab._imp.lp.rng)

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([0, 1], lp, cp)
                self.assertIs(mab._rng, mab._imp.rng)
                self.assertIs(mab._rng, mab._imp.lp_list[0].rng)

    #################################################
    # Test add_arm() method
    ################################################

    def test_add_arm(self):
        for lp in MABTest.lps:
            mab = MAB([0, 1], lp)
            mab.add_arm(2)
            self.assertTrue(2 in mab.arms)
            self.assertTrue(len(mab._imp.arms) == 3)
            self.assertTrue(2 in mab._imp.arm_to_expectation.keys())

            mab.add_arm('a')
            self.assertTrue('a' in mab.arms)
            self.assertTrue(len(mab._imp.arms) == 4)
            self.assertTrue('a' in mab._imp.arm_to_expectation.keys())

    def test_add_arm_contextual(self):
        for lp in MABTest.para_lps:
            mab = MAB([0, 1], lp)
            mab.add_arm(2)
            self.assertTrue(2 in mab.arms)
            self.assertTrue(len(mab._imp.arms) == 3)
            self.assertTrue(2 in mab._imp.arm_to_expectation.keys())

            mab.add_arm('a')
            self.assertTrue('a' in mab.arms)
            self.assertTrue(len(mab._imp.arms) == 4)
            self.assertTrue('a' in mab._imp.arm_to_expectation.keys())

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([0, 1], lp, cp)
                mab.add_arm(2)
                self.assertTrue(2 in mab.arms)
                self.assertTrue(len(mab._imp.arms) == 3)
                self.assertTrue(len(mab._imp.lp.arms) == 3)
                self.assertTrue(2 in mab._imp.lp.arm_to_expectation.keys())

                mab.add_arm('a')
                self.assertTrue('a' in mab.arms)
                self.assertTrue(len(mab._imp.arms) == 4)
                self.assertTrue(len(mab._imp.lp.arms) == 4)
                self.assertTrue('a' in mab._imp.lp.arm_to_expectation.keys())

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([0, 1], lp, cp)
                mab.add_arm(2)
                self.assertTrue(2 in mab.arms)
                self.assertTrue(len(mab._imp.arms) == 3)
                self.assertTrue(len(mab._imp.lp_list[0].arms) == 3)
                self.assertTrue(2 in mab._imp.lp_list[0].arm_to_expectation.keys())

                mab.add_arm('a')
                self.assertTrue('a' in mab.arms)
                self.assertTrue(len(mab._imp.arms) == 4)
                self.assertTrue(len(mab._imp.lp_list[0].arms) == 4)
                self.assertTrue('a' in mab._imp.lp_list[0].arm_to_expectation.keys())

    #################################################
    # Test partial_fit() method
    ################################################

    def test_partial_fit(self):
        for lp in MABTest.lps:
            arm, mab = self.predict(arms=["A", "B", "C", "D"],
                                    decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=lp,
                                    seed=123456,
                                    num_run=1,
                                    is_predict=True)
            mab.partial_fit(["A", "B"], [0, 0])

    def test_partial_fit_contextual(self):
        for lp in MABTest.para_lps:
            arm, mab = self.predict(arms=["A", "B", "C", "D"],
                                    decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=lp,
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    seed=123456,
                                    num_run=1,
                                    is_predict=True)
            mab.partial_fit(["A", "B"], [0, 0], [[1, 3, 1, 1, 1], [0, 0, 0, 0, 0]])

        for cp in MABTest.nps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                arm, mab = self.predict(arms=["A", "B", "C", "D"],
                                        decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                                        rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                         [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                         [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                         [0, 2, 1, 0, 0]],
                                        contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)
                mab.partial_fit(["A", "B"], [0, 0], [[1, 3, 1, 1, 1], [0, 0, 0, 0, 0]])

        for cp in MABTest.cps:
            for lp in MABTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                arm, mab = self.predict(arms=["A", "B", "C", "D"],
                                        decisions=["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"],
                                        rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                        learning_policy=lp,
                                        neighborhood_policy=cp,
                                        context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                         [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                         [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                         [0, 2, 1, 0, 0]],
                                        contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                        seed=123456,
                                        num_run=1,
                                        is_predict=True)
                mab.partial_fit(["A", "B"], [0, 0], [[1, 3, 1, 1, 1], [0, 0, 0, 0, 0]])

    def test_partial_fit_without_fit(self):

        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)
            mab.partial_fit([1, 2], [0, 1])
            x1 = mab.predict()

            mab = MAB([1, 2], lp)
            mab.fit([1, 2], [0, 1])
            x2 = mab.predict()

            self.assertEqual(x1, x2)

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2], para_lp)
            mab.partial_fit([1, 2], [0, 1], [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            x1 = mab.predict([[0, 10, -2, 4, 2]])

            mab = MAB([1, 2], para_lp)
            mab.fit([1, 2], [0, 1], [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])
            x2 = mab.predict([[0, 10, -2, 4, 2]])

            self.assertEqual(x1, x2)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 1, 2, 3, 5],
                                                       [1, 1, 1, 1, 1],
                                                       [0, 0, 0, 0, 0]])
                x1 = mab.predict([[0, 10, -2, 4, 2]])

                mab = MAB([1, 2], lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 1, 2, 3, 5],
                                                       [1, 1, 1, 1, 1],
                                                       [0, 0, 0, 0, 0]])
                x2 = mab.predict([[0, 10, -2, 4, 2]])

                self.assertEqual(x1, x2)

        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2], para_lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0]])

                x1 = mab.predict([[0, 0, 0, 0, 0], ])

                mab = MAB([1, 2], para_lp, cp)
                mab.fit([1, 2, 2], [0, 1, 0], [[0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]])
                x2 = mab.predict([[0, 0, 0, 0, 0]])

                self.assertEqual(x1, x2)

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:
                mab = MAB([1, 2], lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 9, 2, 3, 5],
                                                       [1, 1, 1, 1, 1],
                                                       [-3, 0, 0, -7, 0]])
                x1 = mab.predict([[0, 10, -2, 4, 2]])

                mab = MAB([1, 2], lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 9, 2, 3, 5],
                                                       [1, 1, 1, 1, 1],
                                                       [-3, 0, 0, -7, 0]])
                x2 = mab.predict([[0, 10, -2, 4, 2]])

                self.assertEqual(x1, x2)

        for cp in BaseTest.cps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2], para_lp, cp)
                mab.partial_fit([1, 2, 2], [0, 1, 0], [[0, 9, 2, 3, 5],
                                                       [1, 1, 1, 1, 1],
                                                       [-3, 0, 0, -7, 0]])

                x1 = mab.predict([[0, 0, 0, 0, 0]])

                mab = MAB([1, 2], para_lp, cp)
                mab.fit([1, 2, 2], [0, 1, 0], [[0, 9, 2, 3, 5],
                                               [1, 1, 1, 1, 1],
                                               [-3, 0, 0, -7, 0]])
                x2 = mab.predict([[0, 0, 0, 0, 0]])

                self.assertEqual(x1, x2)

    def test_partial_fit_single_row(self):
        rng = np.random.RandomState(seed=9)
        train_data = pd.DataFrame({'a': [rng.rand() for _ in range(20)],
                                   'b': [rng.rand() for _ in range(20)],
                                   'c': [rng.rand() for _ in range(20)],
                                   'decision': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'reward': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]})
        test_data = pd.DataFrame({'a': [rng.rand() for _ in range(3)], 'b': [rng.rand() for _ in range(3)],
                                  'c': [rng.rand() for _ in range(3)], 'decision': [ 1, 1, 2], 'reward': [ 0, 1, 1]})
        context_columns = ['a', 'b', 'c']

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2], para_lp)
            mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            for index, row in test_data.iterrows():
                mab.predict(row[context_columns])
                mab.partial_fit([row['decision']], [row['reward']], row[context_columns])

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                for index, row in test_data.iterrows():
                    mab.predict(row[context_columns])
                    mab.partial_fit([row['decision']], [row['reward']], row[context_columns])

        # With neighbors based approaches it is difficult to guarantee that
        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2], para_lp, cp)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                for index, row in test_data.iterrows():
                    mab.predict(row[context_columns])
                    mab.partial_fit([row['decision']], [row['reward']], row[context_columns])

    def test_convert_matrix(self):
        a = np.array([[1, 2, 3], [2, 2, 2]])
        b = [[1, 2, 3], [2, 2, 2]]
        c = pd.DataFrame({'one': [1, 2, 3], 'two': [2, 2, 2]})
        d = np.array([[1, 2, 3], [2, 2, 2]], order='F')

        MAB._convert_matrix(None)
        MAB._convert_matrix(a)
        MAB._convert_matrix(b)
        MAB._convert_matrix(c)
        MAB._convert_matrix(c['one'])
        MAB._convert_matrix(c.loc[0], row=True)
        MAB._convert_matrix(d)

    #################################################
    # Test serialization
    ################################################

    def test_pickle_before_fit(self):
        rng = np.random.RandomState(seed=9)
        train_data = pd.DataFrame({'a': [rng.rand() for _ in range(20)],
                                   'b': [rng.rand() for _ in range(20)],
                                   'c': [rng.rand() for _ in range(20)],
                                   'decision': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'reward': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]})
        test_data = pd.DataFrame({'a': [rng.rand() for _ in range(3)], 'b': [rng.rand() for _ in range(3)],
                                  'c': [rng.rand() for _ in range(3)], 'decision': [1, 1, 2], 'reward': [0, 1, 1]})
        context_columns = ['a', 'b', 'c']

        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)
            file = open('mab.pkl', 'wb')
            pickle.dump(mab, file)
            file.close()

            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()

            new_mab.fit(train_data['decision'], train_data['reward'])
            new_mab.predict()

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2], para_lp)
            file = open('mab.pkl', 'wb')
            pickle.dump(mab, file)
            file.close()

            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()

            new_mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            new_mab.predict(test_data[context_columns])

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                file = open('mab.pkl', 'wb')
                pickle.dump(mab, file)
                file.close()

                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()

                new_mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                new_mab.predict(test_data[context_columns])

        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2], para_lp, cp)
                file = open('mab.pkl', 'wb')
                pickle.dump(mab, file)
                file.close()

                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()

                new_mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                new_mab.predict(test_data[context_columns])

        os.remove('mab.pkl')

    def test_pickle_fitted(self):
        rng = np.random.RandomState(seed=9)
        train_data = pd.DataFrame({'a': [rng.rand() for _ in range(20)],
                                   'b': [rng.rand() for _ in range(20)],
                                   'c': [rng.rand() for _ in range(20)],
                                   'decision': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'reward': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]})
        test_data = pd.DataFrame({'a': [rng.rand() for _ in range(3)], 'b': [rng.rand() for _ in range(3)],
                                  'c': [rng.rand() for _ in range(3)], 'decision': [1, 1, 2], 'reward': [0, 1, 1]})
        context_columns = ['a', 'b', 'c']

        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)
            mab.fit(train_data['decision'], train_data['reward'])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            new_mab.predict()

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2], para_lp)
            mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            new_mab.predict(test_data[context_columns])

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                new_mab.predict(test_data[context_columns])

        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2], para_lp, cp)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                new_mab.predict(test_data[context_columns])

        os.remove('mab.pkl')

    def test_pickle_fitted_reproducibility(self):
        rng = np.random.RandomState(seed=9)
        train_data = pd.DataFrame({'a': [rng.rand() for _ in range(20)],
                                   'b': [rng.rand() for _ in range(20)],
                                   'c': [rng.rand() for _ in range(20)],
                                   'decision': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'reward': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]})
        test_data = pd.DataFrame({'a': [rng.rand() for _ in range(3)], 'b': [rng.rand() for _ in range(3)],
                                  'c': [rng.rand() for _ in range(3)], 'decision': [1, 1, 2], 'reward': [0, 1, 1]})
        context_columns = ['a', 'b', 'c']

        for lp in BaseTest.lps:
            mab = MAB([1, 2, 3], lp, seed=11)
            mab.fit(train_data['decision'], train_data['reward'])
            p1 = mab.predict()

            mab2 = MAB([1, 2, 3], lp, seed=11)
            mab2.fit(train_data['decision'], train_data['reward'])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab2, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            p2 = new_mab.predict()

            self.assertEqual(p1, p2)

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2, 3], para_lp, seed=11)
            mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            p1 = mab.predict(test_data[context_columns])

            mab2 = MAB([1, 2, 3], para_lp, seed=11)
            mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab2, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            p2 = new_mab.predict(test_data[context_columns])

            self.assertEqual(p1, p2)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2, 3], lp, cp, seed=11)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                p1 = mab.predict(test_data[context_columns])

                mab2 = MAB([1, 2, 3], lp, cp, seed=11)
                mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab2, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                p2 = new_mab.predict(test_data[context_columns])

                self.assertEqual(p1, p2)

        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2, 3], para_lp, cp, seed=11)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                p1 = mab.predict(test_data[context_columns])

                mab2 = MAB([1, 2, 3], para_lp, cp, seed=11)
                mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab2, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                p2 = new_mab.predict(test_data[context_columns])

                self.assertEqual(p1, p2)

        os.remove('mab.pkl')

    def test_pickle_fitted_reproducibile_expectations(self):
        rng = np.random.RandomState(seed=9)
        train_data = pd.DataFrame({'a': [0.1, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0],
                                   'b': [0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0],
                                   'c': [0, 0.1, 0.1, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0],
                                   'd': [0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1, 0.1, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0],
                                   'decision': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'reward': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]})
        test_data = pd.DataFrame({'a': [0, 0.1, 0],
                                  'b': [0.1, 0, 0],
                                  'c': [0, 0, 0],
                                  'd': [0, 0, 0.1],
                                  'decision': [1, 1, 2],
                                  'reward': [0, 1, 1]})
        context_columns = ['a', 'b', 'c', 'd']

        for lp in BaseTest.lps:
            mab = MAB([1, 2, 3], lp, seed=11)
            mab.fit(train_data['decision'], train_data['reward'])
            pe1 = mab.predict_expectations()

            mab2 = MAB([1, 2, 3], lp, seed=11)
            mab2.fit(train_data['decision'], train_data['reward'])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab2, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            pe2 = new_mab.predict_expectations()

            self.assertDictEqual(pe1, pe2)

        for para_lp in BaseTest.para_lps:
            mab = MAB([1, 2, 3], para_lp, seed=11)
            mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            pe1 = mab.predict_expectations(test_data[context_columns])

            mab2 = MAB([1, 2, 3], para_lp, seed=11)
            mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
            file = open('mab.pkl', 'wb')
            pickle.dump(mab2, file)
            file.close()
            file2 = open('mab.pkl', 'rb')
            new_mab = pickle.load(file2)
            file2.close()
            pe2 = new_mab.predict_expectations(test_data[context_columns])

            self.assertListEqual(pe1, pe2)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2, 3], lp, cp, seed=11)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                pe1 = mab.predict_expectations(test_data[context_columns])

                mab2 = MAB([1, 2, 3], lp, cp, seed=11)
                mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab2, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                pe2 = new_mab.predict_expectations(test_data[context_columns])

                self.assertListEqual(pe1, pe2)

        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:

                if not self.is_compatible(para_lp, cp):
                    continue

                mab = MAB([1, 2, 3], para_lp, cp, seed=11)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                pe1 = mab.predict_expectations(test_data[context_columns])

                mab2 = MAB([1, 2, 3], para_lp, cp, seed=11)
                mab2.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                file = open('mab.pkl', 'wb')
                pickle.dump(mab2, file)
                file.close()
                file2 = open('mab.pkl', 'rb')
                new_mab = pickle.load(file2)
                file2.close()
                pe2 = new_mab.predict_expectations(test_data[context_columns])

                self.assertListEqual(pe1, pe2)

        os.remove('mab.pkl')

    #################################################
    # Test arm_status
    ################################################

    def test_status_add_arm(self):

        cold_arm_status = {'is_trained': False, 'is_warm': False, 'warm_started_by': None}

        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)

            # Add arm
            mab.add_arm(3)
            self.assertEqual(mab._imp.arm_to_status[3], cold_arm_status)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)

                # Add arm
                mab.add_arm(3)
                self.assertEqual(mab._imp.arm_to_status[3], cold_arm_status)

    def test_status_remove_arm(self):

        cold_arm_status = {'is_trained': False, 'is_warm': False, 'warm_started_by': None}

        for lp in BaseTest.lps:
            mab = MAB([1, 2], lp)
            self.assertEqual(mab._imp.arm_to_status[1], cold_arm_status)
            self.assertEqual(mab._imp.arm_to_status[2], cold_arm_status)

            # Remove arm
            mab.remove_arm(1)
            self.assertTrue(1 not in mab._imp.arm_to_status)

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                self.assertEqual(mab._imp.arm_to_status[1], cold_arm_status)
                self.assertEqual(mab._imp.arm_to_status[2], cold_arm_status)

                # Remove arm
                mab.remove_arm(1)
                self.assertTrue(1 not in mab._imp.arm_to_status)

    def test_status_fit(self):

        for lp in MABTest.lps:
            if lp == LearningPolicy.Random():
                continue
            arms, mab = self.predict(arms=[1, 2, 3],
                                     decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                                     rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     learning_policy=lp,
                                     seed=123456,
                                     num_run=1,
                                     is_predict=True)
            self.assertEqual(mab._imp.arm_to_status[1]["is_trained"], True)
            self.assertEqual(mab._imp.arm_to_status[2]["is_trained"], True)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)

        for lp in MABTest.para_lps:
            arms, mab = self.predict(arms=[1, 2, 3],
                                     decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                                     rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     learning_policy=lp,
                                     context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                      [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                      [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                                     contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                     seed=123456,
                                     num_run=1,
                                     is_predict=True)
            self.assertEqual(mab._imp.arm_to_status[1]["is_trained"], True)
            self.assertEqual(mab._imp.arm_to_status[2]["is_trained"], True)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)

    def test_status_warm_start(self):
        for lp in MABTest.lps:
            if lp == LearningPolicy.Random():
                continue
            arms, mab = self.predict(arms=[1, 2, 3],
                                     decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                                     rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     learning_policy=lp,
                                     seed=123456,
                                     num_run=1,
                                     is_predict=True)

            # Before warm start
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)
            self.assertEqual(mab._imp.arm_to_status[3]["is_warm"], False)
            self.assertListEqual(mab.cold_arms, [3])

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)
            self.assertEqual(mab._imp.arm_to_status[3]["is_warm"], True)
            self.assertListEqual(mab.cold_arms, list())

            # Warm start again
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)
            self.assertEqual(mab._imp.arm_to_status[3]["is_warm"], True)
            self.assertListEqual(mab.cold_arms, list())

        for lp in MABTest.para_lps:
            arms, mab = self.predict(arms=[1, 2, 3],
                                     decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                                     rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                     learning_policy=lp,
                                     context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                      [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                      [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]],
                                     contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                     seed=123456,
                                     num_run=1,
                                     is_predict=True)

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)
            self.assertEqual(mab._imp.arm_to_status[3]["is_warm"], True)
            self.assertListEqual(mab.cold_arms, list())

            # Warm start again
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertEqual(mab._imp.arm_to_status[3]["is_trained"], False)
            self.assertEqual(mab._imp.arm_to_status[3]["is_warm"], True)
            self.assertListEqual(mab.cold_arms, list())

    def test_status_fit_warmstart(self):

        for lp in MABTest.lps:
            if lp == LearningPolicy.Random():
                continue
            mab = MAB([1, 2, 3], lp)
            mab.fit(decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1], rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": False, "is_warm": False, "warm_started_by": None})

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": False, "is_warm": True, "warm_started_by": 1})

            # Partial fit
            mab.partial_fit(decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3], rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3], {"is_trained": True, "is_warm": True, "warm_started_by": 1})

            # Fit from scratch
            mab.fit(decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3], rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})

        for lp in MABTest.para_lps:
            mab = MAB([1, 2, 3], lp)
            mab.fit(decisions=[1, 1, 1, 2, 2, 2, 1, 1, 1],
                    rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": False, "is_warm": False, "warm_started_by": None})

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": False, "is_warm": True, "warm_started_by": 1})

            # Partial fit
            mab.partial_fit(decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                            rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                            contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                      [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                      [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3], {"is_trained": True, "is_warm": True, "warm_started_by": 1})

            # Fit from scratch
            mab.fit(decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                    rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                              [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                              [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3]])
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})

            # Warm start
            mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5]}, distance_quantile=0.5)
            self.assertDictEqual(mab._imp.arm_to_status[1],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[2],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
            self.assertDictEqual(mab._imp.arm_to_status[3],
                                 {"is_trained": True, "is_warm": False, "warm_started_by": None})
