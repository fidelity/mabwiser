import unittest
from unittest.mock import patch

import logging

import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from mabwiser.base_mab import BaseMAB
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.simulator import Simulator, _NeighborsSimulator, _RadiusSimulator, _KNearestSimulator, default_evaluator
from mabwiser.greedy import _EpsilonGreedy

logging.disable(logging.CRITICAL)


class TestSimulator(unittest.TestCase):

    lps = [LearningPolicy.EpsilonGreedy(),
           LearningPolicy.Popularity(),
           LearningPolicy.Random(),
           LearningPolicy.UCB1(),
           LearningPolicy.ThompsonSampling(),
           LearningPolicy.Softmax()]

    parametric = [LearningPolicy.LinUCB(), LearningPolicy.LinTS()]

    nps = [NeighborhoodPolicy.LSHNearest(),
           NeighborhoodPolicy.KNearest(),
           NeighborhoodPolicy.Radius(),
           NeighborhoodPolicy.Clusters(),
           NeighborhoodPolicy.TreeBandit()]

    @staticmethod
    def is_compatible(lp, np):

        # Case for TreeBandit lp/np compatibility
        treebandit_compat = isinstance(np, NeighborhoodPolicy.TreeBandit) \
                            and np._is_compatible(lp)

        return treebandit_compat

    def test_contextual_offline(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

    def test_contextual_offline_run(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertEqual(sim._chunk_size, 100)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_contextual_offline_run_n_jobs(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp, n_jobs=2)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para, n_jobs=2)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_contextual_online(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(100)],
                        rewards=[rng.randint(0, 2) for _ in range(100)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_contextual_quick(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7, is_quick=True)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_contextual_online_quick(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(100)],
                        rewards=[rng.randint(0, 2) for _ in range(100)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7, is_quick=True)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_context_free_offline(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 2) for _ in range(10)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

    def test_context_free_offline_run(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 2) for _ in range(10)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_context_free_offline_run_n_jobs(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp, n_jobs=2)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 2) for _ in range(10)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_context_free_online(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=2,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_mixed_offline(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_mixed_online(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(100)],
                        rewards=[rng.randint(0, 2) for _ in range(100)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_contextual_unordered(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=False, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_contextual_unordered_online(self):
        rng = np.random.RandomState(seed=7)
        bandits = []
        counter = 0
        for cp in TestSimulator.nps:
            for lp in TestSimulator.lps:

                if not self.is_compatible(lp, cp):
                    continue

                bandits.append((str(counter), MAB([0, 1], lp, cp)))
                counter += 1

        for para in TestSimulator.parametric:
            bandits.append((str(counter), MAB([0, 1], para)))
            counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(100)],
                        rewards=[rng.randint(0, 2) for _ in range(100)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=False, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_context_free_unordered(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 2) for _ in range(10)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=False, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))

    def test_context_free_unordered_online(self):
        bandits = []
        counter = 0
        for lp in TestSimulator.lps:
            bandits.append((str(counter), MAB([0, 1], lp)))
            counter += 1

        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=None,
                        scaler=None, test_size=0.4, batch_size=2,
                        is_ordered=False, seed=7)
        sim.run()
        self.assertTrue(bool(sim.arm_to_stats_total))
        self.assertTrue(bool(sim.bandit_to_predictions))
        self.assertTrue('total' in sim.bandit_to_arm_to_stats_max['0'].keys())

    def test_contextual_log_file(self):
        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.LinUCB()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7, log_file='test.log')

    def test_contextual_log_format(self):
        rng = np.random.RandomState(seed=7)

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.LinUCB()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7, log_format='%(asctime)s %(message)s')

    @patch("mabwiser.simulator.plt.show")
    def test_plot_avg_arms(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('avg', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_avg_net(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('avg', False)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_min_arms(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('min', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_min_net(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('min', False)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_max_arms(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('max', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_max_net(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('max', False)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_avg_arms_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('avg', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_avg_net_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('avg', False)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_min_arms_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('min', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_min_net_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('min', False)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_max_arms_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('max', True)

    @patch("mabwiser.simulator.plt.show")
    def test_plot_max_net_online(self, mock_show):
        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 100) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=5,
                        is_ordered=True, seed=7)
        sim.run()
        sim.plot('max', False)

    def test_np(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(10)])


        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.LinUCB()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

    def test_expectations_online(self):
        rng = np.random.RandomState(seed=7)
        decisions = pd.Series([rng.randint(0, 2) for _ in range(100)])
        rewards = pd.Series([rng.randint(0, 2) for _ in range(100)])
        contexts = pd.DataFrame([[rng.rand() for _ in range(5)] for _ in range(100)])
        for lp in self.lps:
            sim = Simulator(bandits=[("example", MAB([0, 1], lp))],
                            decisions=decisions,
                            rewards=rewards, test_size=0.4, batch_size=10,
                            is_ordered=True, seed=7)
            sim.run()
            self.assertEqual(len(sim.bandit_to_expectations['example']), 4)

        for par in self.parametric:
            sim = Simulator(bandits=[("example", MAB([0, 1], par))],
                            decisions=decisions,
                            rewards=rewards,
                            contexts=contexts,
                            test_size=0.4, batch_size=10,
                            is_ordered=True, seed=7)
            sim.run()
            self.assertEqual(len(sim.bandit_to_expectations['example']),
                             len(sim.bandit_to_predictions['example']))
            self.assertEqual(len(sim.bandit_to_expectations['example']), 40)

        for nbp in self.nps:
            for lp in self.lps:

                if not self.is_compatible(lp, nbp):
                    continue

                sim = Simulator(bandits=[("example", MAB([0, 1], lp, nbp))],
                                decisions=decisions,
                                rewards=rewards,
                                contexts=contexts,
                                test_size=0.4, batch_size=10,
                                is_ordered=True, seed=7)
                sim.run()

                self.assertEqual(len(sim.bandit_to_expectations['example']),
                                 len(sim.bandit_to_predictions['example']))

                self.assertEqual(len(sim.bandit_to_expectations['example']), 40)

                name, bandit = sim.bandits[0]
                if isinstance(bandit, _NeighborsSimulator):
                    self.assertEqual(len(sim.bandit_to_neighborhood_size['example']),
                                     len(sim.bandit_to_predictions['example']))

                # Skip TreeBandit as it does not work with parametric lps
                if isinstance(nbp, NeighborhoodPolicy.TreeBandit):
                    continue

                for par in self.parametric:

                        sim = Simulator(bandits=[("example", MAB([0, 1], par, nbp))],
                                        decisions=decisions,
                                        rewards=rewards,
                                        contexts=contexts,
                                        test_size=0.4, batch_size=10,
                                        is_ordered=True, seed=7)
                        sim.run()
                        self.assertEqual(len(sim.bandit_to_expectations['example']),
                                         len(sim.bandit_to_predictions['example']))

                        self.assertEqual(len(sim.bandit_to_expectations['example']), 40)

                        name, bandit = sim.bandits[0]
                        if isinstance(bandit, _NeighborsSimulator):
                            self.assertEqual(len(sim.bandit_to_neighborhood_size['example']),
                                             len(sim.bandit_to_predictions['example']))

    def test_expectations_offline(self):
        rng = np.random.RandomState(seed=7)
        decisions = pd.Series([rng.randint(0, 2) for _ in range(100)])
        rewards = pd.Series([rng.randint(0, 2) for _ in range(100)])
        contexts = pd.DataFrame([[rng.rand() for _ in range(5)] for _ in range(100)])
        for lp in self.lps:
            sim = Simulator(bandits=[("example", MAB([0, 1], lp))],
                            decisions=decisions,
                            rewards=rewards, test_size=0.4, batch_size=0,
                            is_ordered=True, seed=7)
            sim.run()

            self.assertEqual(len(sim.bandit_to_expectations['example']), 2)

        for par in self.parametric:
            sim = Simulator(bandits=[("example", MAB([0, 1], par))],
                            decisions=decisions,
                            rewards=rewards,
                            contexts=contexts,
                            test_size=0.4, batch_size=0,
                            is_ordered=True, seed=7)
            sim.run()
            self.assertEqual(len(sim.bandit_to_expectations['example']),
                             len(sim.bandit_to_predictions['example']))
            self.assertEqual(len(sim.bandit_to_expectations['example']), 40)

        for nbp in self.nps:
            for lp in self.lps:

                if not self.is_compatible(lp, nbp):
                    continue

                sim = Simulator(bandits=[("example", MAB([0, 1], lp, nbp))],
                                decisions=decisions,
                                rewards=rewards,
                                contexts=contexts,
                                test_size=0.4, batch_size=0,
                                is_ordered=True, seed=7)
                sim.run()

                self.assertEqual(len(sim.bandit_to_expectations['example']),
                                 len(sim.bandit_to_predictions['example']))

                self.assertEqual(len(sim.bandit_to_expectations['example']), 40)
                name, bandit = sim.bandits[0]
                if isinstance(bandit, _NeighborsSimulator):
                    self.assertEqual(len(sim.bandit_to_neighborhood_size['example']),
                                     len(sim.bandit_to_predictions['example']))

            for par in self.parametric:

                if not self.is_compatible(par, nbp):
                    continue

                if not isinstance(nbp, NeighborhoodPolicy.TreeBandit):
                    sim = Simulator(bandits=[("example", MAB([0, 1], par, nbp))],
                                    decisions=decisions,
                                    rewards=rewards,
                                    contexts=contexts,
                                    test_size=0.4, batch_size=0,
                                    is_ordered=True, seed=7)
                    sim.run()
                    self.assertEqual(len(sim.bandit_to_expectations['example']),
                                     len(sim.bandit_to_predictions['example']))
                    self.assertEqual(len(sim.bandit_to_expectations['example']), 40)

                    name, bandit = sim.bandits[0]
                    if isinstance(bandit, _NeighborsSimulator):
                        self.assertEqual(len(sim.bandit_to_neighborhood_size['example']),
                                         len(sim.bandit_to_predictions['example']))

    def test_context_df(self):
        rng = np.random.RandomState(seed=7)
        decisions = pd.Series([rng.randint(0, 2) for _ in range(10)])
        rewards = pd.Series([rng.randint(0, 100) for _ in range(10)])
        contexts = pd.DataFrame([[rng.rand() for _ in range(5)] for _ in range(10)])

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.LinUCB()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

    def test_context_series(self):
        rng = np.random.RandomState(seed=7)
        decisions = pd.Series([rng.randint(0, 2) for _ in range(10)])
        rewards = pd.Series([rng.randint(0, 100) for _ in range(10)])
        contexts = pd.Series([rng.rand() for _ in range(10)])

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.LinUCB()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

    def test_cold_start(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(10)])

        sim = Simulator(bandits=[("example", MAB([0, 1, 2], LearningPolicy.Softmax()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()

    def test_batch_size_1(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(20)])
        rewards = np.array([rng.randint(0, 100) for _ in range(20)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(20)])

        sim = Simulator(bandits=[("example1", MAB([0, 1], LearningPolicy.Softmax())),
                                 ("example2", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.KNearest())),
                                 ("example3", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(1)))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.4, batch_size=1,
                        is_ordered=True, seed=7)
        sim.run()

    def test_offline_single_prediction(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(20)])
        rewards = np.array([rng.randint(0, 100) for _ in range(20)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(20)])

        sim = Simulator(bandits=[("example1", MAB([0, 1], LearningPolicy.Softmax())),
                                 ("example2", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.KNearest())),
                                 ("example3", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(1)))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=StandardScaler(), test_size=0.05, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()

    def test_custom_evaluator_constructor(self):
        def evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index, nn):
            out = {}
            for arm in arms:
                out[arm] = {'count': 0, 'sum': np.nan, 'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan}
            return out

        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7, evaluator=evaluator)

    def test_custom_evaluator_run(self):
        def evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index, nn):
            out = {}
            for arm in arms:
                out[arm] = {'count': 0, 'sum': np.nan, 'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan}
            return out

        rng = np.random.RandomState(seed=7)
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=[rng.randint(0, 2) for _ in range(10)],
                        rewards=[rng.randint(0, 100) for _ in range(10)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7, evaluator=evaluator)
        sim.run()

    def test_get_stats(self):
        rewards = np.array([1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4])
        stats = Simulator.get_stats(rewards)
        self.assertEqual(stats['count'], 13)
        self.assertEqual(stats['sum'], 35)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)
        self.assertTrue(math.isclose(stats['mean'], 2.692307692, abs_tol=1e-8))
        self.assertTrue(math.isclose(stats['std'], 1.263975132, abs_tol=1e-8))

    def test_get_arm_stats(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        stats = sim.get_arm_stats(decisions, rewards)
        self.assertEqual(list(stats.keys()), [0, 1])
        self.assertEqual(stats[0]['count'], 3)
        self.assertEqual(stats[1]['count'], 7)

    def test_default_evaluator(self):
        rng = np.random.RandomState(seed=7)
        arms = [0, 1, 2]
        decisions = np.array([rng.randint(0, 3) for _ in range(20)])
        rewards = np.array([rng.randint(0, 100) for _ in range(20)])
        predictions = [rng.randint(0, 3) for _ in range(20)]
        arm_to_stats = {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                        1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                        2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}}
        start_index = 0

        stat = 'min'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index)
        self.assertEqual(eval[2]['mean'], 19.5)

        stat = 'max'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index)
        self.assertEqual(eval[2]['mean'], 22.0)

        stat = 'mean'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index)
        self.assertEqual(eval[2]['mean'], 21.0)

    def test_default_evaluator_nn(self):
        rng = np.random.RandomState(seed=7)
        arms = [0, 1, 2]
        decisions = np.array([rng.randint(0, 3) for _ in range(20)])
        rewards = np.array([rng.randint(0, 100) for _ in range(20)])
        predictions = [rng.randint(0, 3) for _ in range(20)]
        train_stats = {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                       1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                       2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}}

        neighborhood_stats = [{0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                               1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {'count': 2, 'sum': 10, 'min': 5, 'max': 5, 'mean': 5, 'std': 0.0},
                               1: {},
                               2: {'count': 5, 'sum': 15, 'min': 0, 'max': 7, 'mean': 3, 'std': 2.5}},
                              {0: {'count': 2, 'sum': 1, 'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.25},
                               1: {'count': 1, 'sum': 4, 'min': 4, 'max': 4, 'mean': 4, 'std': 0.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                               1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                               2: {}},
                              {0: {'count': 3, 'sum': 13, 'min': 3, 'max': 5, 'mean': 4.33, 'std': 1.2},
                               1: {'count': 2, 'sum': 10, 'min': 3, 'max': 7, 'mean': 5, 'std': 1.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {},
                               1: {'count': 1, 'sum': 7, 'min': 7, 'max': 7, 'mean': 7, 'std': 0.0},
                               2: {'count': 4, 'sum': 30, 'min': 7, 'max': 8, 'mean': 7.5, 'std': 0.5}},
                              {0: {'count': 2, 'sum': 10, 'min': 5, 'max': 5, 'mean': 5, 'std': 0.0},
                               1: {},
                               2: {'count': 5, 'sum': 15, 'min': 0, 'max': 7, 'mean': 3, 'std': 2.5}},
                              {0: {'count': 2, 'sum': 1, 'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.25},
                               1: {'count': 1, 'sum': 4, 'min': 4, 'max': 4, 'mean': 4, 'std': 0.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                               1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                               2: {}},
                              {0: {'count': 3, 'sum': 13, 'min': 3, 'max': 5, 'mean': 4.33, 'std': 1.2},
                               1: {'count': 2, 'sum': 10, 'min': 3, 'max': 7, 'mean': 5, 'std': 1.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {},
                               1: {'count': 1, 'sum': 7, 'min': 7, 'max': 7, 'mean': 7, 'std': 0.0},
                               2: {'count': 4, 'sum': 30, 'min': 7, 'max': 8, 'mean': 7.5, 'std': 0.5}},
                              {0: {'count': 2, 'sum': 10, 'min': 5, 'max': 5, 'mean': 5, 'std': 0.0},
                               1: {},
                               2: {'count': 5, 'sum': 15, 'min': 0, 'max': 7, 'mean': 3, 'std': 2.5}},
                              {0: {'count': 2, 'sum': 1, 'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.25},
                               1: {'count': 1, 'sum': 4, 'min': 4, 'max': 4, 'mean': 4, 'std': 0.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                               1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                               2: {}},
                              {0: {'count': 3, 'sum': 13, 'min': 3, 'max': 5, 'mean': 4.33, 'std': 1.2},
                               1: {'count': 2, 'sum': 10, 'min': 3, 'max': 7, 'mean': 5, 'std': 1.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {},
                               1: {'count': 1, 'sum': 7, 'min': 7, 'max': 7, 'mean': 7, 'std': 0.0},
                               2: {'count': 4, 'sum': 30, 'min': 7, 'max': 8, 'mean': 7.5, 'std': 0.5}},
                              {0: {'count': 2, 'sum': 10, 'min': 5, 'max': 5, 'mean': 5, 'std': 0.0},
                               1: {},
                               2: {'count': 5, 'sum': 15, 'min': 0, 'max': 7, 'mean': 3, 'std': 2.5}},
                              {0: {'count': 2, 'sum': 1, 'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.25},
                               1: {'count': 1, 'sum': 4, 'min': 4, 'max': 4, 'mean': 4, 'std': 0.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {'count': 7, 'sum': 21, 'min': 0, 'max': 5, 'mean': 3, 'std': 1.2},
                               1: {'count': 3, 'sum': 15, 'min': 1, 'max': 7, 'mean': 5, 'std': 3.1},
                               2: {}},
                              {0: {'count': 3, 'sum': 13, 'min': 3, 'max': 5, 'mean': 4.33, 'std': 1.2},
                               1: {'count': 2, 'sum': 10, 'min': 3, 'max': 7, 'mean': 5, 'std': 1.0},
                               2: {'count': 5, 'sum': 40, 'min': 5, 'max': 10, 'mean': 8, 'std': 2.5}},
                              {0: {},
                               1: {'count': 1, 'sum': 7, 'min': 7, 'max': 7, 'mean': 7, 'std': 0.0},
                               2: {'count': 4, 'sum': 30, 'min': 7, 'max': 8, 'mean': 7.5, 'std': 0.5}}
                              ]
        arm_to_stats = (train_stats, neighborhood_stats)
        start_index = 1

        stat = 'min'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index, nn=True)
        self.assertEqual(eval[2]['mean'], 20.5)

        stat = 'max'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index, nn=True)
        self.assertEqual(eval[2]['mean'], 21.0)

        stat = 'mean'
        eval = default_evaluator(arms, decisions, rewards, predictions, arm_to_stats, stat, start_index, nn=True)
        self.assertEqual(eval[2]['mean'], 20.75)

    def test_radius_all_empty_neighborhoods(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        contexts = np.array([[rng.rand() for _ in range(20)] for _ in range(10)])

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(1)))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertListEqual(list(set(sim.bandit_to_neighborhood_size['example'])), [0])

    def test_radius_mixed_empty_neighborhoods(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        contexts = np.array([[rng.rand() for _ in range(20)] for _ in range(10)])
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(1.5)))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=contexts,
                        scaler=None, test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        self.assertListEqual(list(set(sim.bandit_to_neighborhood_size['example'])), [0, 2])

    def test_chunk_size(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(10)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertEqual(sim._chunk_size, 100)
        ad, ar, ac, bd, br, bc = sim._run_train_test_split()
        self.assertEqual(sim._chunk_size, len(bd))

        n = 1000000
        decisions = np.array([rng.randint(0, 2) for _ in range(n)])
        rewards = np.array([rng.randint(0, 100) for _ in range(n)])
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy()))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(n)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertEqual(sim._chunk_size, 100)
        ad, ar, ac, bd, br, bc = sim._run_train_test_split()
        self.assertNotEqual(sim._chunk_size, len(bd))
        self.assertEqual(sim._chunk_size, 208)

    def test_negative_n_jobs(self):
        rng = np.random.RandomState(seed=7)
        decisions = np.array([rng.randint(0, 2) for _ in range(100)])
        rewards = np.array([rng.randint(0, 100) for _ in range(100)])
        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy(), n_jobs=2))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertEqual(sim.max_n_jobs, 2)

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy(), n_jobs=-1))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertNotEqual(sim.max_n_jobs, -1)

        sim = Simulator(bandits=[("example", MAB([0, 1], LearningPolicy.EpsilonGreedy(), n_jobs=-1)),
                                 ("example2", MAB([0, 1], LearningPolicy.EpsilonGreedy(), n_jobs=2))],
                        decisions=decisions,
                        rewards=rewards,
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(100)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        self.assertEqual(sim.max_n_jobs, BaseMAB._effective_jobs(40, -1))


    #######################################
    # Neighbors Simulator
    #######################################

    def test_neighbors_simulator_distances(self):
        rng = np.random.RandomState(seed=9)
        nn1 = _NeighborsSimulator(rng, [0, 1], 1, None, _EpsilonGreedy(rng, [0, 1], 1, .05), 'euclidean', True)
        nn2 = _NeighborsSimulator(rng, [0, 1], 1, None, _EpsilonGreedy(rng, [0, 1], 1, .05), 'euclidean', True)
        decisions = np.array([rng.randint(0, 2) for _ in range(5)])
        rewards = np.array([rng.randint(0, 100) for _ in range(5)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(5)])
        nn1.fit(decisions, rewards, contexts)
        nn2.fit(decisions, rewards, contexts)
        new_contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(5)])
        distances = nn1.calculate_distances(contexts=new_contexts)
        self.assertTrue(len(distances) == 5)
        nn2.set_distances(distances=distances)
        self.assertIs(nn1.distances, nn2.distances)

    def test_neighbors_simulator_copy_lp(self):
        rng = np.random.RandomState(seed=7)
        bandits = [('1', MAB([0, 1], LearningPolicy.EpsilonGreedy(), NeighborhoodPolicy.Radius()))]
        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        oname, original = bandits[0]
        nname, new = sim.bandits[0]
        self.assertIs(original._imp.lp, new.lp)

    def test_neighbors_simulator_bandit_replacement(self):
        rng = np.random.RandomState(seed=7)
        nns = [NeighborhoodPolicy.Radius(), NeighborhoodPolicy.KNearest()]

        bandits = []
        counter = 0
        for nn in nns:
            for lp in TestSimulator.lps:
                bandits.append((str(counter), MAB([0, 1], lp, nn)))
                counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)
        sim.run()
        for name, bandit in sim.bandits:
            self.assertTrue(isinstance(bandit, (_RadiusSimulator, _KNearestSimulator)))

    def test_neighbors_simulator_neighborhood_stats(self):
        rng = np.random.RandomState(seed=7)
        nns = [NeighborhoodPolicy.Radius(), NeighborhoodPolicy.KNearest()]

        bandits = []
        counter = 0
        for nn in nns:
            for lp in TestSimulator.lps:
                bandits.append((str(counter), MAB([0, 1], lp, nn)))
                counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        sim.run()
        for name, bandit in sim.bandits:
            self.assertEqual(len(bandit.row_arm_to_expectation), 8)
            self.assertEqual(len(bandit.neighborhood_arm_to_stat), 8)

    def test_neighbors_simulator_neighborhood_stats_binarizer(self):
        rng = np.random.RandomState(seed=7)

        def binarize(decision, reward):

            if decision == 0:
                return reward <= 50
            else:
                return reward >= 220

        bandits = [('0', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize), NeighborhoodPolicy.Radius())),
                   ('1', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize), NeighborhoodPolicy.KNearest()))]

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 1000) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        sim.run()
        for name, bandit in sim.bandits:
            self.assertEqual(len(bandit.row_arm_to_expectation), 8)
            self.assertEqual(len(bandit.neighborhood_arm_to_stat), 8)

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 1000) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=3,
                        is_ordered=True, seed=7)

        sim.run()
        for name, bandit in sim.bandits:
            self.assertEqual(len(bandit.row_arm_to_expectation), 8)
            self.assertEqual(len(bandit.neighborhood_arm_to_stat), 8)

    def test_neighbors_simulator_neighborhood_size(self):
        rng = np.random.RandomState(seed=7)
        nns = [NeighborhoodPolicy.Radius()]

        bandits = []
        counter = 0
        for nn in nns:
            for lp in TestSimulator.lps:
                bandits.append((str(counter), MAB([0, 1], lp, nn)))
                counter += 1

        sim = Simulator(bandits=bandits,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 2) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        sim.run()
        for name, bandit in sim.bandits:
            self.assertEqual(len(bandit.neighborhood_sizes), 8)

    def test_neighbors_simulator_predict_expectations(self):
        rng = np.random.RandomState(seed=9)
        nn1 = _RadiusSimulator(rng, [0, 1], 1, None, _EpsilonGreedy(rng, [0, 1], 1, .05), radius=2,
                               metric='euclidean', is_quick=True)
        nn2 = _KNearestSimulator(rng, [0, 1], 1, None, _EpsilonGreedy(rng, [0, 1], 1, .05), k=3,
                                 metric='euclidean', is_quick=True)
        decisions = np.array([rng.randint(0, 2) for _ in range(5)])
        rewards = np.array([rng.randint(0, 100) for _ in range(5)])
        contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(5)])
        nn1.fit(decisions, rewards, contexts)
        nn2.fit(decisions, rewards, contexts)
        new_contexts = np.array([[rng.rand() for _ in range(5)] for _ in range(5)])
        distances = nn1.calculate_distances(contexts=new_contexts)

        nn2.set_distances(distances=distances)

        exp1 = nn1.predict_expectations(new_contexts)
        exp2 = nn2.predict_expectations(new_contexts)
        self.assertTrue(isinstance(exp1, list))
        self.assertTrue(isinstance(exp2, list))
        new_single_context = np.array([[rng.rand() for _ in range(5)]])
        exp3 = nn1.predict_expectations(new_single_context)
        exp4 = nn2.predict_expectations(new_single_context)
        self.assertTrue(isinstance(exp3, dict))
        self.assertTrue(isinstance(exp4, dict))

    def test_neighbors_simulator_expectations_no_neighbors(self):
        rng = np.random.RandomState(seed=9)
        nn1 = _RadiusSimulator(rng, [0, 1], 1, None, _EpsilonGreedy(rng, [0, 1], 1, .05), radius=1,
                               metric='euclidean', is_quick=True)
        decisions = np.array([rng.randint(0, 2) for _ in range(10)])
        rewards = np.array([rng.randint(0, 100) for _ in range(10)])
        contexts = np.array([[rng.rand() for _ in range(20)] for _ in range(10)])
        nn1.fit(decisions, rewards, contexts)
        new_contexts = np.array([[rng.rand() for _ in range(20)] for _ in range(5)])

        nn1.calculate_distances(contexts=new_contexts)

        exp1 = nn1.predict_expectations(new_contexts)

        self.assertTrue(isinstance(exp1, list))

        new_single_context = np.array([[rng.rand() for _ in range(20)]])
        exp3 = nn1.predict_expectations(new_single_context)

        self.assertTrue(isinstance(exp3, dict))
        self.assertTrue(np.isnan(exp3[0]))
        self.assertTrue(np.isnan(exp3[1]))

    def test_radius_simulator_empty_neighborhood_custom_prob(self):
        empty_nbhd = [0.25, 0.75]

        rng = np.random.RandomState(seed=9)

        lp = _EpsilonGreedy(rng, [0, 1], 1, .05)

        bandit = [('example', MAB([0, 1], LearningPolicy.EpsilonGreedy(0),
                                  NeighborhoodPolicy.Radius(radius=1, no_nhood_prob_of_arm=empty_nbhd)))]

        sim = Simulator(bandits=bandit,
                        decisions=[rng.randint(0, 2) for _ in range(20)],
                        rewards=[rng.randint(0, 1000) for _ in range(20)],
                        contexts=[[rng.rand() for _ in range(5)] for _ in range(20)],
                        scaler=StandardScaler(), test_size=0.4, batch_size=0,
                        is_ordered=True, seed=7)

        sim._set_stats("total", sim.decisions, sim.rewards)

        train_decisions, train_rewards, train_contexts, test_decisions, test_rewards, test_contexts = \
            sim._run_train_test_split()

        sim._set_stats("train", train_decisions, train_rewards)
        sim._set_stats("test", test_decisions, test_rewards)

        sim._train_bandits(train_decisions, train_rewards, train_contexts)

        test_bandit = sim.bandits[0][1]
        out = [test_bandit._get_no_nhood_predictions(lp, True) for i in range(5)]
        self.assertIsInstance(test_bandit, _RadiusSimulator)
        self.assertListEqual(empty_nbhd, test_bandit.no_nhood_prob_of_arm)
        self.assertListEqual(out, [0, 1, 1, 1, 0])

