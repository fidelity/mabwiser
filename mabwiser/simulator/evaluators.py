# -*- coding: utf-8 -*-

"""
This module provides a utility for evaluating rewqrds.
"""
from abc import ABC, abstractmethod

import math

from typing import List, Dict

import numpy as np


class BaseEvaluator(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def evaluator(
            arms: List[str],
            decisions: np.ndarray,
            rewards: np.ndarray,
            predictions: List[str],
            arm_to_stats: dict,
            stat: str,
            start_index: int,
            nn: bool = False,
    ):
        """The simulator supports custom evaluation functions, but they must have this signature to work with the
        simulation pipeline."""
        pass


class DefaultEvaluator(BaseEvaluator):
    def __init__(self):
        super(DefaultEvaluator, self).__init__()

    @staticmethod
    def evaluator(
            arms: List[str],
            decisions: np.ndarray,
            rewards: np.ndarray,
            predictions: List[str],
            arm_to_stats: dict,
            stat: str,
            start_index: int,
            nn: bool = False,
    ) -> Dict:
        """Default evaluation function.

            Calculates predicted rewards for the test batch based on predicted arms.
            When the predicted arm is the same as the historic decision, the historic reward is used.
            When the predicted arm is different, the mean, min or max reward from the training data is used.
            If using Radius or KNearest neighborhood policy, the statistics from the neighborhood are used
            instead of the entire training set.

            The simulator supports custom evaluation functions,
            but they must have this signature to work with the simulation pipeline.

            Parameters
            ----------
            arms: list
                The list of arms.
            decisions: np.ndarray
                The historic decisions for the batch being evaluated.
            rewards: np.ndarray
                The historic rewards for the batch being evaluated.
            predictions: list
                The predictions for the batch being evaluated.
            arm_to_stats: dict
                The dictionary of descriptive statistics for each arm to use in evaluation.
            stat: str
                Which metric from arm_to_stats to use. Takes the values 'min', 'max', 'mean'.
            start_index: int
                The index of the first row in the batch.
                For offline simulations it is 0.
                For _online simulations it is batch size * batch number.
                Used to select the correct index from arm_to_stats if there are separate entries for each row in the test set.
            nn: bool
                Whether the results are from one of the simulator custom nearest neighbors implementations.

            Returns
            -------
            An arm_to_stats dictionary for the predictions in the batch.
            Dictionary has the format {arm {'count', 'sum', 'min', 'max', 'mean', 'std'}}
            """
        # If decision and prediction matches each other, use the observed reward
        # If decision and prediction are different, use the given stat (e.g., mean) for the arm as the reward

        arm_to_rewards = dict((arm, []) for arm in arms)
        if nn:
            arm_to_stats, neighborhood_stats = arm_to_stats
        for index, predicted_arm in enumerate(predictions):

            if predicted_arm == decisions[index]:
                arm_to_rewards[predicted_arm].append(rewards[index])
            elif nn:
                nn_index = index + start_index
                row_neighborhood_stats = neighborhood_stats[nn_index]
                if row_neighborhood_stats and row_neighborhood_stats[predicted_arm]:
                    arm_to_rewards[predicted_arm].append(
                        row_neighborhood_stats[predicted_arm][stat]
                    )
                else:
                    arm_to_rewards[predicted_arm].append(arm_to_stats[predicted_arm][stat])

            else:
                arm_to_rewards[predicted_arm].append(arm_to_stats[predicted_arm][stat])

        # Calculate stats based on the rewards from predicted arms
        arm_to_stats_prediction = {}
        for arm in arms:
            arm_to_rewards[arm] = np.array(arm_to_rewards[arm])
            if len(arm_to_rewards[arm]) > 0:
                arm_to_stats_prediction[arm] = {
                    "count": arm_to_rewards[arm].size,
                    "sum": arm_to_rewards[arm].sum(),
                    "min": arm_to_rewards[arm].min(),
                    "max": arm_to_rewards[arm].max(),
                    "mean": arm_to_rewards[arm].mean(),
                    "std": arm_to_rewards[arm].std(),
                }
            else:
                arm_to_stats_prediction[arm] = {
                    "count": 0,
                    "sum": math.nan,
                    "min": math.nan,
                    "max": math.nan,
                    "mean": math.nan,
                    "std": math.nan,
                }

        return arm_to_stats_prediction
