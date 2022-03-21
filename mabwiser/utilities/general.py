# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides various general utilities
"""

import multiprocessing as mp
from typing import Dict

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
