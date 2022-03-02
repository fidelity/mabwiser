# -*- coding: utf-8 -*-

import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tests.test_base import BaseTest

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.simulator import Simulator


logging.disable(logging.CRITICAL)


class ExampleTest(BaseTest):

    def test_popularity(self):

        list_of_arms = ['Arm1', 'Arm2']
        decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
        rewards = [20, 17, 25, 9]
        mab = MAB(list_of_arms, LearningPolicy.Popularity())
        mab.fit(decisions, rewards)
        mab.predict()
        self.assertEqual("Arm1", mab.predict())
        self.assertDictEqual({'Arm1': 0.9674354610872193, 'Arm2': 0.032564538912780716},
                             mab.predict_expectations())

    def test_random(self):

        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                learning_policy=LearningPolicy.Random(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 12, 7, 19]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        mab.add_arm(3)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_greedy15(self):

        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 12, 7, 19]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        mab.add_arm(3)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_linucb(self):
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinUCB(alpha=1.25),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [5, 2])

        mab.partial_fit(decisions=arms, rewards=test_df_revenue, contexts=test)

        mab.add_arm(6)
        self.assertTrue(6 in mab.arms)
        self.assertTrue(6 in mab._imp.arm_to_expectation.keys())

    def test_softmax(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                learning_policy=LearningPolicy.Softmax(tau=1),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 12, 7, 19]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        mab.add_arm(3)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_lints(self):
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinTS(alpha=1.5),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [5, 2])

        mab.partial_fit(decisions=arms, rewards=test_df_revenue, contexts=test)

        mab.add_arm(6)
        self.assertTrue(6 in mab.arms)
        self.assertTrue(6 in mab._imp.arm_to_expectation.keys())

    def test_ts(self):

        dec_to_threshold = {1: 10, 2: 20}

        def binarize(dec, value):
            return value >= dec_to_threshold[dec]

        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                learning_policy=LearningPolicy.ThompsonSampling(binarizer=binarize),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 12, 7, 19]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        # Updating of the model with new arm
        def binary_func2(decision, reward):
            if decision == 3:
                return 1 if reward > 15 else 0
            else:
                return 1 if reward > 10 else 0

        mab.add_arm(3, binary_func2)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_ts_binary(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                                learning_policy=LearningPolicy.ThompsonSampling(),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 1)

        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 1, 0, 1]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        mab.add_arm(3)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_ucb1(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=[1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                                rewards=[10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)
        layout_partial = [1, 2, 1, 2]
        revenue_partial = [0, 12, 7, 19]

        mab.partial_fit(decisions=layout_partial, rewards=revenue_partial)

        mab.add_arm(3)
        self.assertTrue(3 in mab.arms)
        self.assertTrue(3 in mab._imp.arm_to_expectation.keys())

    def test_ts_series(self):

        df = pd.DataFrame({'layouts': [1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                           'revenues': [10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]})

        arm, mab = self.predict(arms=[1, 2],
                                decisions=df['layouts'],
                                rewards=df['revenues'],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

    def test_ts_numpy(self):
        arm, mab = self.predict(arms=[1, 2],
                                decisions=np.array([1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1]),
                                rewards=np.array([10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]),
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
                                seed=123456,
                                num_run=1,
                                is_predict=True)

        self.assertEqual(arm, 2)

    def test_approximate(self):
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_tables=5, n_dimensions=5),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [1, 4])

        mab.partial_fit(decisions=arms, rewards=test_df_revenue, contexts=test)

        mab.add_arm(6)
        self.assertTrue(6 in mab.arms)
        self.assertTrue(6 in mab._imp.arm_to_expectation.keys())

    def test_radius(self):
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(radius=5),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [4, 4])

        mab.partial_fit(decisions=arms, rewards=test_df_revenue, contexts=test)

        mab.add_arm(6)
        self.assertTrue(6 in mab.arms)
        self.assertTrue(6 in mab._imp.arm_to_expectation.keys())

    def test_nearest(self):
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.UCB1(alpha=1.25),
                                 neighborhood_policy=NeighborhoodPolicy.KNearest(k=5),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [5, 1])

        mab.partial_fit(decisions=arms, rewards=test_df_revenue, contexts=test)

        mab.add_arm(6)
        self.assertTrue(6 in mab.arms)
        self.assertTrue(6 in mab._imp.arm_to_expectation.keys())

    def test_linucb_radius(self):

        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinUCB(alpha=1.25),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(radius=1),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [1, 2])

    def test_linucb_knearest(self):

        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinUCB(alpha=1.25),
                                 neighborhood_policy=NeighborhoodPolicy.KNearest(k=4),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [1, 2])

    def test_lints_radius(self):

        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinTS(alpha=0.5),
                                 neighborhood_policy=NeighborhoodPolicy.Radius(radius=1),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [1, 2])

    def test_lints_knearest(self):

        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})

        # Scale the data
        scaler = StandardScaler()
        train = scaler.fit_transform(np.asarray(train_df[['age', 'click_rate', 'subscriber']], dtype='float64'))
        test = scaler.transform(np.asarray(test_df, dtype='float64'))

        arms, mab = self.predict(arms=[1, 2, 3, 4, 5],
                                 decisions=train_df['ad'],
                                 rewards=train_df['revenues'],
                                 learning_policy=LearningPolicy.LinTS(alpha=1),
                                 neighborhood_policy=NeighborhoodPolicy.KNearest(k=4),
                                 context_history=train,
                                 contexts=test,
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, [1, 2])

    def test_simulator_contextual(self):
        size = 100

        decisions = [random.randint(0, 2) for _ in range(size)]
        rewards = [random.randint(0, 1000) for _ in range(size)]
        contexts = [[random.random() for _ in range(50)] for _ in range(size)]

        def binarize(decision, reward):

            if decision == 0:
                return reward <= 50
            else:
                return reward >= 220

        n_jobs = 1
        contextual_mabs = [('Random', MAB([0, 1], LearningPolicy.Random(), NeighborhoodPolicy.Radius(10),
                                          n_jobs=n_jobs)),
                           ('UCB1', MAB([0, 1], LearningPolicy.UCB1(1), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                           ('ThompsonSampling', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize),
                                                    NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                           ('EpsilonGreedy', MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=.15),
                                                 NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                           ('Softmax', MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(10),
                                           n_jobs=n_jobs))]

        sim = Simulator(contextual_mabs, decisions, rewards, contexts,
                        scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456)
        sim.run()

        self.assertTrue(sim.bandit_to_confusion_matrices)
        self.assertTrue(sim.bandit_to_predictions)

    def test_simulator_context_free(self):
        size = 100

        decisions = [random.randint(0, 2) for _ in range(size)]
        rewards = [random.randint(0, 1000) for _ in range(size)]

        def binarize(decision, reward):

            if decision == 0:
                return reward <= 50
            else:
                return reward >= 220

        n_jobs = 1
        context_free_mabs = [('Random', MAB([0, 1], LearningPolicy.Random(), n_jobs=n_jobs)),
                             ('UCB1', MAB([0, 1], LearningPolicy.UCB1(1), n_jobs=n_jobs)),
                             ('ThompsonSampling', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize),
                                                      n_jobs=n_jobs)),
                             ('EpsilonGreedy', MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=.15), n_jobs=n_jobs)),
                             ('Softmax', MAB([0, 1], LearningPolicy.Softmax(), n_jobs=n_jobs))]

        sim = Simulator(context_free_mabs, decisions, rewards, contexts=None,
                        scaler=None, test_size=0.5, is_ordered=False, batch_size=1, seed=123456)
        sim.run()

        self.assertTrue(sim.bandit_to_confusion_matrices)
        self.assertTrue(sim.bandit_to_predictions)

    def test_simulator_mixed(self):
        size = 100

        decisions = [random.randint(0, 2) for _ in range(size)]
        rewards = [random.randint(0, 1000) for _ in range(size)]
        contexts = [[random.random() for _ in range(50)] for _ in range(size)]

        n_jobs = 1
        mixed = [('RandomRadius', MAB([0, 1], LearningPolicy.Random(), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                 ('Random', MAB([0, 1], LearningPolicy.Random(), n_jobs=n_jobs))]

        sim = Simulator(mixed, decisions, rewards, contexts,
                        scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456)
        sim.run()

        self.assertTrue(sim.bandit_to_confusion_matrices)
        self.assertTrue(sim.bandit_to_predictions)

    def test_simulator_hyper_parameter(self):
        size = 100

        decisions = [random.randint(0, 2) for _ in range(size)]
        rewards = [random.randint(0, 1000) for _ in range(size)]
        contexts = [[random.random() for _ in range(50)] for _ in range(size)]

        n_jobs = 1
        hyper_parameter_tuning = []
        for radius in range(6, 10):
            hyper_parameter_tuning.append(('Radius' + str(radius),
                                           MAB([0, 1], LearningPolicy.UCB1(1), NeighborhoodPolicy.Radius(radius),
                                               n_jobs=n_jobs)))

        sim = Simulator(hyper_parameter_tuning, decisions, rewards, contexts,
                        scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456,
                        is_quick=True)
        sim.run()

        self.assertTrue(sim.bandit_to_confusion_matrices)
        self.assertTrue(sim.bandit_to_predictions)

    def test_treebandit(self):
        # Arms
        ads = [1, 2, 3, 4, 5]

        # Historical data of ad decisions with corresponding revenues and context information
        train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                                 'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                                 'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                                 'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                                0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                                 'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                                )

        # Test data to for new prediction
        test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
        test_df_revenue = pd.Series([7, 13])

        # Scale the training and test data
        scaler = StandardScaler()
        train = scaler.fit_transform(train_df[['age', 'click_rate', 'subscriber']].values.astype('float64'))
        test = scaler.transform(test_df.values.astype('float64'))

        # TreeBandit contextual policy with DecisionTreeClassifier's default tree parameters
        # and ucb1 learning with alpha of 1.25
        treebandit = MAB(arms=ads,
                         learning_policy=LearningPolicy.UCB1(alpha=1.25),
                         neighborhood_policy=NeighborhoodPolicy.TreeBandit())

        # Learn from previous ads shown and revenues generated
        treebandit.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

        # Predict the next best ad to show
        prediction = treebandit.predict(test)

        # Expectation of each ad based on learning from past ad revenues
        expectations = treebandit.predict_expectations(test)

        # Results
        # print("TreeBandit: ", prediction, " ", expectations)
        assert (prediction == [4, 4])

        # Online update of model
        treebandit.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

        # Updating of the model with new arm
        treebandit.add_arm(6)