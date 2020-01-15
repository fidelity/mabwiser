# -*- coding: utf-8 -*-
# SPDX-License-Identifer: Apache-2.0

"""
:Author: FMR LLC
:Email: mabwiser@fmr.com

This module provides a number of constants and helper functions.
"""

import numpy as np

from typing import Dict, Union, NamedTuple, NewType, NoReturn

Arm = NewType('Arm', Union[int, float, str])
"""Arm type is defined as integer, float, or string."""

Num = Union[int, float]
"""Num type is defined as integer or float."""


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    default_seed = 123456
    """The default random seed."""

    distance_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                        "euclidean", "hamming", "jaccard", "kulsinski", "mahalanobis", "matching", "minkowski",
                        "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean"]
    """The distance metrics supported by neighborhood policies."""


class RandomGenerator:
    def __init__(self, seed):

        self.rng = np.random.RandomState(seed=seed)

        # Note: To change to use np.random.Generator
        # self.rng = np.random.default_rng(seed=seed_sequence.generate_state(1)[0])

    @staticmethod
    def get_random_generator(seed):
        rng_generator = RandomGenerator(seed)
        return rng_generator.rng

    @staticmethod
    def get_state(rng):
        return rng.get_state()[1][0]


def argmax(dictionary: Dict[Arm, Num]) -> Arm:
    """
    Returns the first key with the maximum value.
    """
    return max(dictionary, key=dictionary.get)


def check_false(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def reset(dictionary: Dict, value) -> NoReturn:
    """
    Maps every key to the given value.
    """
    dictionary.update({}.fromkeys(dictionary, value))

