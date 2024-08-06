# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging

from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from tests.test_base import BaseTest

from mabwiser.base_mab import BaseMAB
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.simulator import Simulator


logging.disable(logging.CRITICAL)


class InvalidTest(BaseTest):

    def test_invalid_arm_not_list(self):
        with self.assertRaises(TypeError):
            MAB(1, LearningPolicy.EpsilonGreedy(epsilon=0))

    def test_invalid_learning_policy(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], NeighborhoodPolicy.Radius(radius=12))

    def test_incomplete_learning_policy_implementation(self):
        class TestMAB(BaseMAB):
            def __init__(self):
                rng = np.random.RandomState(7)
                arms = [0,1]
                n_jobs = 1
                backend = None
                super().__init__(rng, arms, n_jobs, backend)

            def _fit_arm(self, arm, decisions, rewards, contexts=None):
                pass

            def _predict_contexts(self, contexts, is_predict, seeds=None, start_index=None):
                pass

            def _uptake_new_arm(self, arm, binarizer=None, scaler=None):
                pass

            def _drop_existing_arm(self, arm):
                pass

            def _copy_arms(self, cold_arm_to_warm_arm):
                pass

            def fit(self, decisions, rewards, contexts=None):
                pass

            def partial_fit(self, decisions, rewards, contexts=None):
                pass

            def predict(self, contexts=None):
                pass

            def predict_expectations(self, contexts=None):
                pass

            def warm_start(self, arm_to_features, distance_quantile):
                pass

        mab = MAB([0, 1], learning_policy=LearningPolicy.EpsilonGreedy())
        mab._imp = TestMAB()
        with self.assertRaises(NotImplementedError):
            mab.learning_policy

    def test_invalid_epsilon_type(self):
        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.EpsilonGreedy(epsilon="one"))

    def test_invalid_epsilon_value(self):
        with self.assertRaises(ValueError):
            MAB(['a', 'b'], LearningPolicy.EpsilonGreedy(epsilon=2))
        with self.assertRaises(ValueError):
            MAB(['a', 'b'], LearningPolicy.LinGreedy(epsilon=-1))
        with self.assertRaises(ValueError):
            MAB(['a', 'b'], LearningPolicy.LinGreedy(epsilon=2))

    def test_invalid_rewards_to_binary_type(self):
        thresholds = {1: 1, 'b': 1}
        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.ThompsonSampling(thresholds))

    def test_invalid_ridge_alpha_value(self):
        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinUCB(alpha=-1),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinTS(alpha=0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_ridge_alpha_type(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinUCB(alpha=None),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinTS(alpha=None),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_ridge_l2_lambda_value(self):
        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=-1),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinTS(alpha=1, l2_lambda=0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_ridge_l2_lambda_type(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinGreedy(l2_lambda=None),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinUCB(alpha=1, l2_lambda=None),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.LinTS(alpha=1, l2_lambda=None),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_ucb_alpha_type(self):
        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.UCB1(alpha="one"))

    def test_invalid_ucb_alpha_value(self):
        with self.assertRaises(ValueError):
            MAB(['a', 'b'], LearningPolicy.UCB1(alpha=-2))

    def test_invalid_tau_type(self):
        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.Softmax(tau="one"))

    def test_invalid_tau_value(self):
        with self.assertRaises(ValueError):
            MAB(['a', 'b'], LearningPolicy.Softmax(tau=0))

    def test_invalid_lp_arg(self):
        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.UCB1(epsilon=2))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.EpsilonGreedy(alpha=2))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.ThompsonSampling(alpha=2))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.Softmax(alpha=2))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.LinGreedy(alpha=1))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.LinUCB(tau=1))

        with self.assertRaises(TypeError):
            MAB(['a', 'b'], LearningPolicy.LinTS(epsilon=1))

    def test_invalid_context_policy(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), LearningPolicy.EpsilonGreedy(epsilon=0))

    def test_invalid_n_tables_type(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0),
                NeighborhoodPolicy.LSHNearest(n_tables='string'))

    def test_invalid_n_tables_value(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0),
                NeighborhoodPolicy.LSHNearest(n_tables=0))

    def test_invalid_n_dimensions_type(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0),
                NeighborhoodPolicy.LSHNearest(n_dimensions='string'))

    def test_invalid_n_dimensions_value(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0),
                NeighborhoodPolicy.LSHNearest(n_dimensions=0))

    def test_invalid_radius_no_nhood_type_ann(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.LSHNearest(no_nhood_prob_of_arm={}))

    def test_invalid_radius_no_nhood_sum_ann(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(no_nhood_prob_of_arm=[0, 0]))

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(metric='linear'))

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(radius=-1))

    def test_invalid_radius_no_nhood_type(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(radius=1,
                                                                                           no_nhood_prob_of_arm={}))

    def test_invalid_radius_no_nhood_sum(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(radius=1,
                                                                                           no_nhood_prob_of_arm=[0, 0]))

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.KNearest(k=0))

    def test_too_large_k(self):
        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(4),
                         context_history=[[1, 1], [0, 0], [0, 0]],
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_minibatch(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Clusters(minibatch=0))

    def test_invalid_clusters_type(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Clusters(n_clusters=None))

    def test_invalid_clusters_num(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Clusters(n_clusters=1))

    def test_invalid_treebandit_lp_linucb(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.LinUCB(), NeighborhoodPolicy.TreeBandit())

    def test_invalid_treebandit_lp_popularity(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.Popularity(), NeighborhoodPolicy.TreeBandit())

    def test_invalid_treebandit_lp_softmax(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.TreeBandit())

    def test_invalid_treebandit_lp_lints(self):
        with self.assertRaises(ValueError):
            MAB([0, 1], LearningPolicy.LinTS(), NeighborhoodPolicy.TreeBandit())

    def test_invalid_seed(self):
        with self.assertRaises(TypeError):
            MAB([0, 1], LearningPolicy.EpsilonGreedy(0), seed=[0, 1])

    def test_predict_with_no_fit(self):
        for lp in InvalidTest.lps:
            mab = MAB([1, 2], lp)
            with self.assertRaises(Exception):
                mab.predict_expectations()

        for lp in InvalidTest.para_lps:
            mab = MAB([1, 2], lp)
            with self.assertRaises(Exception):
                mab.predict_expectations([[0, 1, 1, 2]])

        for cp in InvalidTest.nps:
            for lp in InvalidTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                mab = MAB([1, 2], lp, cp)
                with self.assertRaises(Exception):
                    mab.predict_expectations([[0, 1, 1, 2]])

    # Context
    def test_invalid_context_single(self):

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                with self.assertRaises(TypeError):
                    self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1],
                                 rewards=[0],
                                 learning_policy=lp,
                                 neighborhood_policy=cp,
                                 context_history=[0, 1, 2, 3, 5],
                                 contexts=[0, 1, 2, 3, 5],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

    def test_context_single_history(self):

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                with self.assertRaises(TypeError):
                    self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1],
                                 rewards=[0],
                                 learning_policy=lp,
                                 neighborhood_policy=cp,
                                 context_history=[0],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

    def test_context_single_context(self):

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:

                if not self.is_compatible(lp, cp):
                    continue

                with self.assertRaises(TypeError):
                    self.predict(arms=[1, 2, 3, 4],
                                 decisions=np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                                 rewards=np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                                 learning_policy=lp,
                                 neighborhood_policy=cp,
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[0, 1, 2, 3, 5],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

    # Fit
    def test_invalid_decisions_rewards_length(self):
        decisions = [1, 1, 2, 2, 2, 3, 3]
        rewards = [0, 0, 0, 0, 0, 0, 1, 1, 1]
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(ValueError):
            mab.fit(decisions, rewards)

    def test_invalid_context_length(self):
        decisions = [1, 1, 1]
        rewards = [0, 0, 0]
        context_history = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(2))
        with self.assertRaises(ValueError):
            mab.fit(decisions, rewards, context_history)

    def test_invalid_context_type(self):
        decisions = [1, 1, 1]
        rewards = [0, 0, 0]
        context_history = {1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1], 4: [1, 1, 1]}
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(2))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards, context_history)

    def test_rewards_null_list(self):
        decisions = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        rewards = [0, 0, 0, 0, 0, 0, 1, 1, None]
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards)

    def test_rewards_null_array(self):
        decisions = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3])
        rewards = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, None])
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards)

    def test_rewards_nan_array(self):
        decisions = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3])
        rewards = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, np.nan])
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards)

    def test_rewards_null_df(self):
        history = pd.DataFrame({'decision': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                'reward': [0, 0, 0, 0, 0, 0, 1, 1, None]})
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(history['decision'], history['reward'])

    def test_rewards_inf_array(self):
        decisions = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3])
        rewards = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, np.inf])
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards)

    def test_rewards_inf_df(self):
        history = pd.DataFrame({'decision': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                'reward': [0, 0, 0, 0, 0, 0, 1, 1, np.inf]})
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(history['decision'], history['reward'])

    def test_invalid_no_context_policy(self):
        decisions = [1, 1, 1]
        rewards = [0, 0, 0]
        context_history = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards, context_history)

    def test_invalid_no_context_history(self):
        decisions = [1, 1, 1]
        rewards = [0, 0, 0]
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0), NeighborhoodPolicy.Radius(2))
        with self.assertRaises(TypeError):
            mab.fit(decisions, rewards)

    def test_invalid_2d_context_list(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=[[1, 1], [0, 0], [0, 0]],
                         contexts=[1, 1],
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_2d_context_np(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=[[1, 1], [0, 0], [0, 0]],
                         contexts=np.array([1, 1]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_2d_context_history_list(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=[1, 1, 1],
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_2d_context_history_np(self):
        with self.assertRaises(TypeError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1],
                         rewards=[0, 0, 0],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.0),
                         neighborhood_policy=NeighborhoodPolicy.KNearest(2),
                         context_history=np.array([1, 1, 1]),
                         contexts=np.array([[1, 1]]),
                         seed=123456,
                         num_run=1,
                         is_predict=True)

    def test_invalid_add_arm(self):
        mab = MAB([1, 2, 3], LearningPolicy.EpsilonGreedy(epsilon=0))
        with self.assertRaises(ValueError):
            mab.add_arm(None)
        with self.assertRaises(ValueError):
            mab.add_arm(np.nan)
        with self.assertRaises(ValueError):
            mab.add_arm(np.inf)
        with self.assertRaises(ValueError):
            mab.add_arm(3)

    def test_exps_without_fit(self):
        for lp in BaseTest.lps:
            with self.assertRaises(Exception):
                mab = MAB([1, 2], lp)
                mab.predict_expectations()

        for para_lp in BaseTest.para_lps:
            with self.assertRaises(Exception):
                mab = MAB([1, 2], para_lp)
                mab.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:
                with self.assertRaises(Exception):
                    mab = MAB([1, 2], lp, cp)
                    mab.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        for cp in BaseTest.nps:
            for para_lp in BaseTest.lps:
                with self.assertRaises(Exception):
                    mab = MAB([1, 2], para_lp, cp)
                    mab.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

    def test_predict_without_fit(self):
        for lp in BaseTest.lps:
            with self.assertRaises(Exception):
                mab = MAB([1, 2], lp)
                mab.predict()

        for para_lp in BaseTest.para_lps:
            with self.assertRaises(Exception):
                mab = MAB([1, 2], para_lp)
                mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        for cp in BaseTest.nps:
            for lp in BaseTest.lps:
                with self.assertRaises(Exception):
                    mab = MAB([1, 2], lp, cp)
                    mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        for cp in BaseTest.nps:
            for para_lp in BaseTest.lps:
                with self.assertRaises(Exception):
                    mab = MAB([1, 2], para_lp, cp)
                    mab.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

    def test_invalid_jobs(self):
        with self.assertRaises(ValueError):
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 3, 2, 2, 3, 1, 3],
                         rewards=[0, 1, 1, 0, 1, 0, 1, 1, 1],
                         learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                         seed=123456,
                         num_run=4,
                         is_predict=True,
                         n_jobs=0)

    def test_invalid_scale(self):
        with self.assertRaises(TypeError):
            mab = MAB([0, 1], LearningPolicy.LinUCB(scale=1))

    def test_convert_array_invalid(self):
        df = pd.DataFrame({'a': [1, 1, 1, 1, 1]})
        with self.assertRaises(NotImplementedError):
            MAB._convert_array(df)

    def test_convert_matrix_invalid(self):
        data = {'one': [1, 1, 1, 1, 1]}
        with self.assertRaises(NotImplementedError):
            MAB._convert_matrix(data)

    ####################################################################
    # Simulator
    ####################################################################

    def test_invalid_plot_args_metric(self):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                    decisions=[rng.randint(0, 2) for _ in range(10)],
                    rewards=[rng.randint(0, 100) for _ in range(10)],
                    contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                    scaler=StandardScaler(), test_size=0.4, batch_size=0,
                    is_ordered=True, seed=7)
        sim.run()
        with self.assertRaises(TypeError):
            sim.plot(metric=1)

    def test_invalid_plot_args_metric_value(self):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                    decisions=[rng.randint(0, 2) for _ in range(10)],
                    rewards=[rng.randint(0, 100) for _ in range(10)],
                    contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                    scaler=StandardScaler(), test_size=0.4, batch_size=0,
                    is_ordered=True, seed=7)
        sim.run()
        with self.assertRaises(ValueError):
            sim.plot('mean')

    def test_invalid_plot_args_per_arm(self):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                    decisions=[rng.randint(0, 2) for _ in range(10)],
                    rewards=[rng.randint(0, 100) for _ in range(10)],
                    contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                    scaler=StandardScaler(), test_size=0.4, batch_size=0,
                    is_ordered=True, seed=7)
        sim.run()
        with self.assertRaises(TypeError):
            sim.plot(is_per_arm=1)

    def test_invalid_not_run_plot(self):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        with self.assertRaises(AssertionError):
            sim.plot()

    def test_invalid_get_stats(self):
        data = np.array(['h','e','l','l','o'])
        with self.assertRaises(TypeError):
            Simulator.get_stats(data)

    def test_invalid_get_arm_stats(self):
        rng = np.random.RandomState(seed=9)
        decisions = np.array([rng.randint(0, 2) for _ in range(5)])
        rewards = np.array([rng.randint(0, 100) for _ in range(5)])
        new_rewards = np.array(['h','e','l','l','o'])

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=decisions,
                        rewards= rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(5)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        with self.assertRaises(TypeError):
            stats = sim.get_arm_stats(decisions, new_rewards)

    def test_invalid_rewards_simulator(self):
        rng = np.random.RandomState(seed=7)

        with self.assertRaises(ValueError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(9)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=.4, batch_size=0,
                      is_ordered=True, seed=7)

    def test_invalid_contexts_simulator(self):
        rng = np.random.RandomState(seed=7)

        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[{1: rng.rand(), 2: rng.rand()} for _ in range(10)],
                      scaler=StandardScaler(), test_size=.4, batch_size=0,
                      is_ordered=True, seed=7)

    def test_invalid_bandits(self):

        rng = np.random.RandomState(seed=7)

        with self.assertRaises(TypeError):
            Simulator(bandits=[MAB([0, 1], LearningPolicy.EpsilonGreedy())],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=1, batch_size=0,
                      is_ordered=True, seed=7)

    def test_invalid_test_size(self):
        rng = np.random.RandomState(seed=7)

        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=1, batch_size=0,
                      is_ordered=True, seed=7)

        with self.assertRaises(ValueError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=50.0, batch_size=0,
                      is_ordered=True, seed=7)

    def test_invalid_batch_size(self):
        rng = np.random.RandomState(seed=7)

        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0.5,
                      is_ordered=True, seed=7)

        with self.assertRaises(ValueError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=10,
                      is_ordered=True, seed=7)

    def test_invalid_is_ordered(self):
        rng = np.random.RandomState(seed=7)
        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0,
                      is_ordered=1, seed=7)

    def test_invalid_evaluator_function(self):
        rng = np.random.RandomState(seed=7)
        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0,
                      is_ordered=True, seed=7, evaluator=50)

    def test_invalid_quick_run(self):
        rng = np.random.RandomState(seed=7)
        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0, is_quick=1,
                      is_ordered=True, seed=7)

    def test_invalid_log_file(self):
        rng = np.random.RandomState(seed=7)
        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0,
                      is_ordered=True, seed=7, log_file=7)

    def test_invalid_log_format(self):
        rng = np.random.RandomState(seed=7)
        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0,
                      is_ordered=True, seed=7, log_format=7)

        with self.assertRaises(TypeError):
            Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                      decisions=[rng.randint(0, 2) for _ in range(10)],
                      rewards=[rng.randint(0, 100) for _ in range(10)],
                      contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                      scaler=StandardScaler(), test_size=0.4, batch_size=0,
                      is_ordered=True, seed=7, log_format=None)

    def test_invalid_simulator_stats_scope(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        with self.assertRaises(ValueError):
            sim._set_stats('validation', decisions, rewards)

    def test_invalid_warm_start_features(self):
        mab = MAB(arms=[1, 2, 3], learning_policy=LearningPolicy.EpsilonGreedy())
        with self.assertRaises(TypeError):
            mab.warm_start([0.1, 0.2, 0.3], 0.5)

    def test_invalid_warm_start_features_none(self):
        mab = MAB(arms=[1, 2, 3], learning_policy=LearningPolicy.EpsilonGreedy())
        with self.assertRaises(TypeError):
            mab.warm_start(None, 0.5)

    def test_invalid_warm_start_quantile_value(self):
        mab = MAB(arms=[1, 2, 3], learning_policy=LearningPolicy.EpsilonGreedy())
        with self.assertRaises(ValueError):
            mab.warm_start({1: [1, 0.5], 2: [1, 0], 3: [0.2, 0.5]}, 50.)

    def test_invalid_warm_start_quantile_none(self):
        mab = MAB(arms=[1, 2, 3], learning_policy=LearningPolicy.EpsilonGreedy())
        with self.assertRaises(TypeError):
            mab.warm_start({1: [1, 0.5], 2: [1, 0], 3: [0.2, 0.5]}, None)
