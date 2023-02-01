# -*- coding: utf-8 -*-

import numpy as np
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.approximate import _LSHNearest
from tests.test_base import BaseTest


class LSHNearestTest(BaseTest):

    def test_hash_function(self):
        seed = 11
        n_dimensions = 5
        n_tables = 5
        table_to_plane = {i: [] for i in range(n_tables)}
        rng = np.random.RandomState(seed)
        contexts = np.array([[rng.rand() for _ in range(7)] for _ in range(10)])

        table_to_plane = {i: rng.standard_normal(size=(contexts.shape[1],
                                                       n_dimensions))
                          for i in table_to_plane.keys()}

        hash_lists = []
        for k in table_to_plane.keys():
            hashes = _LSHNearest.get_context_hash(contexts, table_to_plane[k])
            hash_lists.append(list(hashes))
        self.assertListEqual(hash_lists[0], [22, 14, 11, 14, 18,  9,  6,  7, 14, 11])
        self.assertListEqual(hash_lists[1], [28, 29, 21, 7, 20, 28,  4, 28, 28, 20])
        self.assertListEqual(hash_lists[2], [3, 19, 19, 18, 19, 19, 19, 19, 19, 19])
        self.assertListEqual(hash_lists[3], [1, 1, 9, 16, 1, 11, 1, 7, 3, 1])
        self.assertListEqual(hash_lists[4], [18, 16, 20, 26, 18, 19, 20, 16, 16, 16])

    def test_hash_function2(self):
        seed = 11
        n_dimensions = 5
        n_tables = 1
        table_to_plane = {i: [] for i in range(n_tables)}
        rng = np.random.RandomState(seed)
        contexts = np.array([[1, 1, 1, 1, 1, 1, 1],      # Expected same hash
                             [1, 0.9, 1, 1, 1, 1, 0.9],  # Expected same hash
                             [1, 1, 1, 1, 1, 1, 0],      # Expected same hash
                             [-1, 0, 1, 0, -1, 0, 0]     # Expected different hash
                             ])

        table_to_plane = {i: rng.standard_normal(size=(contexts.shape[1],
                                                       n_dimensions))
                          for i in table_to_plane.keys()}

        hash_lists = []
        for k in table_to_plane.keys():
            hashes = _LSHNearest.get_context_hash(contexts, table_to_plane[k])
            hash_lists.append(list(hashes))
        self.assertListEqual(hash_lists[0], [5, 5, 5, 14])

    def test_tables(self):
        seed = 11
        n_dimensions = 5
        n_tables = 5
        rng = np.random.default_rng(seed)
        contexts = np.array([[rng.random() for _ in range(7)] for _ in range(10)])
        decisions = np.array([rng.integers(0, 2) for _ in range(10)])
        rewards = np.array([rng.random() for _ in range(10)])
        lsh = MAB(arms=[0, 1], learning_policy=LearningPolicy.Softmax(),
                  neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions, n_tables),
                  seed=seed)
        for i in range(n_tables):
            self.assertListEqual([], lsh._imp.table_to_plane[i])

        lsh.fit(decisions, rewards, contexts)
        self.assertListAlmostEqual(list(lsh._imp.table_to_plane[0][0]),
                                   [0.03419276725318417, 1.3597475403099617, 1.2247210785859324,
                                    -0.5103070767876675, -0.2979695111064471])
        self.assertListEqual(list(lsh._imp.table_to_hash_to_index[0].keys()), [0, 1, 2, 3, 7, 11])
        self.assertListEqual(lsh._imp.table_to_hash_to_index[0][1], [2, 3])
        self.assertListEqual(lsh._imp.table_to_hash_to_index[0][14], [])

    def test_partial_fit_indices(self):
        seed = 11
        n_dimensions = 5
        n_tables = 5
        rng = np.random.RandomState(seed)
        contexts = np.array([[rng.rand() for _ in range(7)] for _ in range(10)])
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.rand() for _ in range(10)])
        lsh = MAB(arms=[0, 1], learning_policy=LearningPolicy.Softmax(),
                  neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions, n_tables),
                  seed=seed)
        lsh.fit(decisions, rewards, contexts)
        contexts2 = np.array([[rng.rand() for _ in range(7)] for _ in range(10)])
        decisions2 = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards2 = np.array([rng.rand() for _ in range(10)])
        lsh.partial_fit(decisions2, rewards2, contexts2)

        self.assertListEqual(lsh._imp.table_to_hash_to_index[0][4], [])
        self.assertListEqual(lsh._imp.table_to_hash_to_index[0][12], [])

    def test_greedy0_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 3])

    def test_greedy0_d2_single_test(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, 1)

    def test_greedy0_d2_single_list(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertEqual(arms, 1)

    def test_greedy0_d2_exps(self):

        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        self.assertDictEqual(exps[0], {1: 0.6666666666666666, 2: 0.0, 3: 0.6, 4: 0})
        self.assertDictEqual(exps[1], {1: 0.6666666666666666, 2: 0.0, 3: 0.75, 4: 0})

    def test_greedy0_d5(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=5),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 4, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 5, 4, 3, 0], [0, 1, 5, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [2, 2])

    def test_greedy1_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=1.0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 4])

    def test_thompson_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.ThompsonSampling(),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 1])

    def test_ucb_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.UCB1(alpha=1),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 1])

    def test_softmax_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.Softmax(tau=1),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 3])

    def test_no_neighbors_hash(self):
        contexts = [[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1], [0, -1, -2, -3, -5], [-1, -1, -1, -1, -1],
                                           [0, -1, -2, -3, -5]]
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=20, n_tables=1),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=contexts,
                                 seed=7,
                                 num_run=1,
                                 is_predict=True)

        hashes = mab._imp.get_context_hash(np.asarray(contexts), mab._imp.table_to_plane[0])
        for h in hashes:
            self.assertEqual(len(mab._imp.table_to_hash_to_index[0][h]), 0)

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=20, n_tables=1),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=contexts,
                                 seed=12,
                                 num_run=1,
                                 is_predict=True)

        hashes = mab._imp.get_context_hash(np.asarray(contexts), mab._imp.table_to_plane[0])
        for h in hashes:
            self.assertEqual(len(mab._imp.table_to_hash_to_index[0][h]), 0)

    def test_no_neighbors(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=20, n_tables=1),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1], [0, -1, -2, -3, -5], [-1, -1, -1, -1, -1],
                                           [0, -1, -2, -3, -5]],
                                 seed=7,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [4, 3, 4, 3, 4])

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=20, n_tables=1),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1], [0, -1, -2, -3, -5], [-1, -1, -1, -1, -1],
                                           [0, -1, -2, -3, -5]],
                                 seed=12,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [3, 3, 1, 3, 2])

    def test_no_neighbors_expectations(self):

        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[10, 10, 10, -10, -10, -10],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=25),
                                context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                 [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1]],
                                seed=7,
                                num_run=1,
                                is_predict=False)

        for index, row in enumerate(exp):
            for key in row.keys():
                self.assertIs(np.nan, row[key])

        exp, mab = self.predict(arms=[1, 2, 3],
                                decisions=[1, 1, 1, 2, 2, 2],
                                rewards=[10, 10, 10, -10, -10, -10],
                                learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=25),
                                context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1]],
                                seed=7,
                                num_run=1,
                                is_predict=False)

        for index, row in enumerate(exp):
            for key in row.keys():
                self.assertIs(np.nan, row[key])

    def test_partial_fit_greedy0_d2(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertListEqual(arms, [1, 3])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)

        decisions2 = [1, 2, 3]
        rewards2 = [1, 1, 1]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 13)
        self.assertEqual(len(mab._imp.rewards), 13)
        self.assertEqual(len(mab._imp.contexts), 13)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)

    def test_partial_fit_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp.is_contextual_binarized)
        self.assertListEqual(arms, [3, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.partial_fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 13)
        self.assertEqual(len(mab._imp.rewards), 13)
        self.assertEqual(len(mab._imp.contexts), 13)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

    def test_fit_twice_thompson_thresholds(self):

        arm_to_threshold = {1: 1, 2: 5, 3: 2, 4: 3}

        def binarize(arm, reward):
            return reward >= arm_to_threshold[arm]

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 7, 0, 1, 9, 0, 2, 6, 11],
                                 learning_policy=LearningPolicy.ThompsonSampling(binarize),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)

        self.assertTrue(mab._imp.lp.is_contextual_binarized)
        self.assertListEqual(arms, [3, 1])
        self.assertEqual(len(mab._imp.decisions), 10)
        self.assertEqual(len(mab._imp.rewards), 10)
        self.assertEqual(len(mab._imp.contexts), 10)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

        decisions2 = [1, 2, 3]
        rewards2 = [11, 1, 6]
        context_history2 = [[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
        mab.fit(decisions2, rewards2, context_history2)

        self.assertEqual(len(mab._imp.decisions), 3)
        self.assertEqual(len(mab._imp.rewards), 3)
        self.assertEqual(len(mab._imp.contexts), 3)
        self.assertEqual(np.ndim(mab._imp.decisions), 1)
        self.assertTrue(mab._imp.rewards.all() in [0, 1])

    def test_add_arm(self):

        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
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
        self.assertTrue(5 in mab._imp.lp.arms)
        self.assertTrue(5 in mab._imp.lp.arm_to_expectation.keys())

    def test_greedy0_no_nhood_predict_random(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force to no nhood
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=25),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1]],
                                 seed=7,
                                 num_run=2,
                                 is_predict=True)

        # 3rd arm was never seen but picked up by random neighborhood in both tests
        self.assertListEqual(arms[0], [3, 3])
        self.assertListEqual(arms[1], [1, 1])

    def test_greedy0_no_nhood_predict_weighted(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force to no nhoods
        arms, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(
                                     n_dimensions=25, no_nhood_prob_of_arm=[0, 0.8, 0.2]),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1]],
                                 seed=7,
                                 num_run=2,
                                 is_predict=True)

        # 2nd arm is weighted highly but 3rd is picked too
        self.assertListEqual(arms[0], [3, 2])
        self.assertListEqual(arms[1], [2, 2])

    def test_greedy0_no_nhood_expectation_nan(self):

        # 2nd, 3rd arm has bad rewards should not be selected
        # Use small neighborhood size to force to no nhoods
        exps, mab = self.predict(arms=[1, 2, 3],
                                 decisions=[1, 1, 1, 2, 2, 2],
                                 rewards=[10, 10, 10, -10, -10, -10],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=25),
                                 context_history=[[1, 1, 2, 3, 5], [1, 2, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0]],
                                 contexts=[[0, -1, -2, -3, -5], [-1, -1, -1, -1, -1]],
                                 seed=7,
                                 num_run=1,
                                 is_predict=False)

        # When there are no neighborhoods, expectations will be nan
        self.assertDictEqual(exps[0], {1: np.nan, 2: np.nan, 3: np.nan})
        self.assertDictEqual(exps[1], {1: np.nan, 2: np.nan, 3: np.nan})

    def test_remove_arm(self):
        arms, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=True)
        self.assertTrue(mab.arms is mab._imp.arms)
        self.assertTrue(mab.arms is mab._imp.lp.arms)
        mab.remove_arm(3)
        self.assertTrue(3 not in mab.arms)
        self.assertTrue(3 not in mab._imp.arms)
        self.assertTrue(3 not in mab._imp.lp.arms)
        self.assertTrue(3 not in mab._imp.lp.arm_to_expectation)

    def test_warm_start(self):
        exps, mab = self.predict(arms=[1, 2, 3, 4],
                                 decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                 rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                 learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                 neighborhood_policy=NeighborhoodPolicy.LSHNearest(n_dimensions=2),
                                 context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                  [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                  [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                  [0, 2, 1, 0, 0]],
                                 contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                 seed=123456,
                                 num_run=1,
                                 is_predict=False)

        # Before warm start
        self.assertDictEqual(exps[0], {1: 0.6666666666666666, 2: 0.0, 3: 0.6, 4: 0})
        self.assertDictEqual(exps[1], {1: 0.6666666666666666, 2: 0.0, 3: 0.75, 4: 0})

        # Warm start
        mab.warm_start(arm_to_features={1: [0, 1], 2: [0, 0], 3: [0.5, 0.5], 4: [0, 1]}, distance_quantile=0.5)
        exps = mab.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]])

        # After warm start
        self.assertDictEqual(exps[0], {1: 0.6666666666666666, 2: 0.0, 3: 0.6, 4: 0})
        self.assertDictEqual(exps[1], {1: 0.6666666666666666, 2: 0.0, 3: 0.75, 4: 0})
