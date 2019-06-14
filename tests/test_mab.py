# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class MABTest(BaseTest):

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

                self.assertEqual(np.ndim(mab._imp.contexts), 2)

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:
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

                self.assertEqual(np.ndim(mab._imp.contexts), 2)

        for cp in BaseTest.cps:
            for lp in BaseTest.lps:
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
        self.assertEqual(arms, [3, 3, 3, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [2, 3, 3, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.25),
                                 seed=79,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 3, 2])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=123456,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 3, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=7,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [2, 1, 1, 3])
        self.assertIs(mab._rng, mab._imp.rng)

        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 rewards=[0, 0, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.33),
                                 seed=79,
                                 num_run=4,
                                 is_predict=True)

        self.assertEqual(len(arms), 4)
        self.assertEqual(arms, [3, 3, 3, 2])
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
                mab = MAB([0, 1], lp, cp)
                self.assertIs(mab._rng, mab._imp.rng)
                self.assertIs(mab._rng, mab._imp.lp.rng)

        for cp in MABTest.cps:
            for lp in MABTest.lps:
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
                mab = MAB([1, 2], lp, cp)
                mab.fit(train_data['decision'], train_data['reward'], train_data[context_columns])
                for index, row in test_data.iterrows():
                    mab.predict(row[context_columns])
                    mab.partial_fit([row['decision']], [row['reward']], row[context_columns])

        # With neighbors based approaches it is difficult to guarantee that
        for cp in BaseTest.nps:
            for para_lp in BaseTest.para_lps:
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

