# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides various general utilities
"""

import multiprocessing as mp
from typing import Dict
import numpy as np

from mabwiser.utilities.types import Num


def effective_jobs(size: int, n_jobs: int) -> int:
    if n_jobs < 0:
        n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
    n_jobs = min(n_jobs, size)
    return n_jobs


def argmax(dictionary: Dict[str, Num]) -> str:
    """
    Returns the first key with the maximum value.
    """
    return max(dictionary, key=dictionary.get)


def argmin(dictionary: Dict) -> str:
    """
    Returns the first key that has the minimum value.
    """
    return min(dictionary, key=dictionary.get)


def reset(dictionary: Dict, value) -> None:
    """
    Maps every key to the given value.
    """
    dictionary.update({}.fromkeys(dictionary, value))


def get_stats(rewards: np.ndarray) -> Dict:
    """Calculates descriptive statistics for the given array of rewards.

    Parameters
    ----------
    rewards: nd.nparray
        Array of rewards for a single arm.

    Returns
    -------
    A dictionary of descriptive statistics.
    Dictionary has the format {'count', 'sum', 'min', 'max', 'mean', 'std'}
    """
    return {
        "count": rewards.size,
        "sum": rewards.sum(),
        "min": rewards.min(),
        "max": rewards.max(),
        "mean": rewards.mean(),
        "std": rewards.std(),
    }


def get_context_hash(contexts, plane):
    # Project rows onto plane and get signs
    projection_signs = 1 * (np.dot(contexts, plane) > 0)

    # Get base 2 value of projection signs
    # Another approach is to convert to strings ('01000', '00101', '11111', etc)
    hash_values = np.zeros(contexts.shape[0])
    for i in range(plane.shape[1]):
        hash_values = hash_values + (projection_signs[:, i] * 2**i)

    return hash_values
