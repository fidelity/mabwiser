# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import List

from mabwiser.configs.learning import LearningPolicy


def check_sum_to_unity(list_floats: List[float]) -> None:
    if isinstance(list_floats, (list, List)):
        if not np.isclose(sum(list_floats), 1.0):
            raise ValueError(
                f"The list of values should sum up close to 1.0 -- currently `{sum(list_floats)}`"
            )


def is_compatible(learning_policy: LearningPolicy):
    # TreeBandit is compatible with these learning policies
    return isinstance(learning_policy, (LearningPolicy.EpsilonGreedy,
                                        LearningPolicy.UCB1,
                                        LearningPolicy.ThompsonSampling))
